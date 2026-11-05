"""Genuine, lazy FLARE-LLM Large inference. Never substitutes simulated captions."""
from __future__ import annotations
import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    pass


class FlareModel:
    name = 'FLARE-LLM Large'
    supports_visual_grounding = True
    # Live drafts and final calls execute the same recognition code.
    supports_exact_result_reuse = True

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self._search = None
        self._lock = threading.Lock()
        self._loading = False
        self._error = None
        self.beam = int(os.getenv('FLARE_BEAM_SIZE', '30'))
        if not 1 <= self.beam <= 30:
            raise ValueError('FLARE_BEAM_SIZE must be between 1 and 30.')
        self.assets = Path(os.getenv('FLARE_MODEL_DIR', str(ROOT / 'models'))).resolve()
        base = self.assets / 'FLARE-checkpoints'
        self.pretrained = base / 'exp_22_2_large/checkpoints/checkpoint_52_800000.pt'
        self.finetuned = base / 'exp_22_2_large/finetuned/checkpoints/checkpoint_last.pt'
        self.adaptor = base / 'lipreading_exp_2_large_face_lip_s2/checkpoints/checkpoint_4_60000.pt'
        self.qwen = self.assets / 'Qwen2.5-7B'
        self.runtime = ROOT / 'vendor/flare_runtime'

    def status(self) -> dict:
        missing = [str(p.relative_to(self.assets)) for p in [self.pretrained, self.finetuned, self.adaptor, self.qwen / 'config.json', self.qwen / 'model.safetensors.index.json', self.assets / '20words_mean_face.npy'] if not p.is_file()]
        if not (self.assets / 'visual_speech/finetuning_TalkSet.model').is_file():
            missing.append('visual speech checkpoint (run scripts/setup_visual_speech.py)')
        if not (ROOT / 'vendor/light-asd/model/Encoder.py').is_file():
            missing.append('visual speech source (run scripts/setup_visual_speech.py)')
        index = self.qwen / 'model.safetensors.index.json'
        if index.is_file():
            import json
            try:
                for name in set(json.loads(index.read_text())['weight_map'].values()):
                    if not (self.qwen / name).is_file():
                        missing.append('Qwen2.5-7B/' + name)
            except (ValueError, KeyError, TypeError):
                missing.append('valid Qwen shard index')
        if not (self.runtime / 'SOURCE.json').is_file():
            missing.append('prepared FLARE source (run scripts/setup_model.py)')
        missing_packages = [p for p in ['torch', 'fairseq', 'transformers', 'peft', 'face_alignment'] if importlib.util.find_spec(p) is None]
        reason = self._error
        if not reason:
            if any(item.startswith('visual speech') for item in missing):
                reason = 'Visual speech filter assets are missing. Run scripts/setup_visual_speech.py in the model environment.'
            elif missing:
                reason = 'Model assets are not installed. Run scripts/setup_model.py --download in the model environment.'
            elif missing_packages or sys.version_info >= (3, 11):
                reason = 'Start the server with .venv-model/bin/python (Python 3.10 and model dependencies).'
            else:
                reason = 'Loading model…' if self._loading else 'Model is installed; load it to start captioning.'
        ready = self.model is not None and not self._loading
        visual = getattr(self.preprocessor, 'visual_speech', None)
        return {'name': self.name, 'model': self.name, 'device': 'CUDA', 'ready': ready, 'state': 'ready' if ready else ('loading' if self._loading else 'unavailable'), 'detail': 'Visual-only inference ready.' if ready else reason, 'missing_assets': missing, 'missing_packages': missing_packages, 'beam_size': self.beam, 'fps': 25, 'native_streaming': False,
                'speech_detector': {'name': 'Light-ASD visual', 'checkpoint': 'talkset',
                                    'ready': visual is not None and not self._loading}}

    def load(self):
        with self._lock:
            if self.model is not None:
                return
            self._error = None
            state = self.status()
            if state['missing_assets'] or state['missing_packages'] or sys.version_info >= (3, 11):
                self._error = state['detail']
                raise ModelUnavailable(self._error)
            self._loading = True
            try:
                import torch
                if not torch.cuda.is_available():
                    raise ModelUnavailable('FLARE-LLM Large requires a CUDA GPU for this runtime.')
                free, total = torch.cuda.mem_get_info()
                minimum = 28 * 1024**3
                if free < minimum:
                    raise ModelUnavailable(f'GPU has {free / 1024**3:.1f} GiB free; reserve at least 28 GiB for FLARE-LLM Large. Existing GPU processes were left untouched.')
                if not torch.cuda.is_bf16_supported():
                    raise ModelUnavailable('The GPU must support BF16 inference.')
                self._load_weights(torch)
                self._warmup(torch)
            except Exception as exc:
                self.model = None
                self.preprocessor = None
                self._error = str(exc)
                LOGGER.warning('FLARE model load failed: %s', exc)
                # Drop the failed loader's frames before releasing cached allocations;
                # otherwise traceback locals can retain a partially loaded GPU model.
                exc.__traceback__ = None
                import gc
                gc.collect()
                torch_module = sys.modules.get('torch')
                if torch_module is not None and torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                raise ModelUnavailable(self._error) from None
            finally:
                self._loading = False

    def _warmup(self, torch):
        """Pay one-time compilation costs before declaring the GPU ready."""
        started = time.perf_counter()
        self.preprocessor.warmup()
        with torch.inference_mode():
            # Camera cadence often crosses the 2 s boundary with 52 samples.
            # Warm both common initial shapes, without padding real requests.
            for count in (51, 52):
                video = torch.zeros((1, 1, count, 88, 88), device='cuda', dtype=torch.bfloat16)
                sample = {'net_input': {'source': {'audio': None, 'video': (video, video),
                            'face_lip_mask': (torch.ones(1, device='cuda'), torch.ones(1, device='cuda'))},
                            'padding_mask': torch.zeros((1, count), dtype=torch.bool, device='cuda')}}
                # Discarded synthetic warmup, never a user transcript.
                self._search([self.model], sample, beam_size=self.beam,
                             audio_only=False, video_only=True, temp=1.2)
        torch.cuda.synchronize()
        LOGGER.info('Model and preprocessing warmup complete in %.2fs', time.perf_counter() - started)

    def _load_weights(self, torch, device="cuda"):
        from omegaconf import OmegaConf, open_dict
        sys.path.insert(0, str(self.runtime.parent))
        from flare_runtime.models.hubert_distill_frm_whisper_offline_bestrq_dual_frontend import AVHubertConfig, AVHubertModelDual
        from flare_runtime.models.hubert_asr_chime_s3_llm import AVHubertSeq2SeqLLMConfig, AVHubertSeq2SeqLLM_chime_s3_llm, HubertEncoderWrapper
        # These pickle checkpoints are solely the explicitly downloaded official release.
        visual = torch.load(self.pretrained, map_location='cpu', weights_only=False)
        cfg = OmegaConf.create(visual['cfg'])
        cfg.model = OmegaConf.merge(OmegaConf.structured(AVHubertConfig), cfg.model)
        with open_dict(cfg.model):
            cfg.model.resnet_weights = None  # all frontend weights come from the release
        with open_dict(cfg.task):
            if 'sample_rate' not in cfg.task:
                cfg.task.sample_rate = 16000
        stats = visual['model'].get('whisper_stat')
        if stats is None:
            raise ModelUnavailable('Released visual checkpoint is missing Whisper normalization statistics.')
        # Upstream requires a .npy training-stat file during construction. Recover the
        # actual buffer from its checkpoint instead of using a researcher-local path.
        import numpy as np
        with tempfile.TemporaryDirectory(prefix='flare-stat-') as tmp:
            path = Path(tmp) / 'whisper.npy'
            np.save(path, stats.cpu().numpy())
            os.environ['FLARE_WHISPER_STAT'] = str(path)
            encoder = AVHubertModelDual(cfg.model, cfg.task, [])
        encoder.load_state_dict(visual['model'], strict=True)
        del visual
        tuned = torch.load(self.finetuned, map_location='cpu', weights_only=False)
        prefix = 'encoder.w2v_model.'
        weights = {key[len(prefix):]: value for key, value in tuned['model'].items() if key.startswith(prefix)}
        if not weights:
            raise ModelUnavailable('Finetuned visual checkpoint has no encoder weights.')
        encoder.load_state_dict(weights, strict=True)
        del tuned, weights
        encoder.remove_pretraining_modules()
        checkpoint = torch.load(self.adaptor, map_location='cpu', weights_only=False)
        full_cfg = OmegaConf.create(checkpoint['cfg'])
        full_cfg.model = OmegaConf.merge(OmegaConf.structured(AVHubertSeq2SeqLLMConfig), full_cfg.model)
        with open_dict(full_cfg.model):
            full_cfg.model.pretrained_path = ''
            full_cfg.model.pretrained_encoder_path = ''
        os.environ['FLARE_QWEN_PATH'] = str(self.qwen)
        model = AVHubertSeq2SeqLLM_chime_s3_llm(HubertEncoderWrapper(encoder), full_cfg.model)
        required = {name for name, _ in model.named_parameters() if 'lora_' in name or name.startswith('modality_projection.')}
        missing = required - checkpoint['model'].keys()
        if missing:
            raise ModelUnavailable(f'Incomplete adapter checkpoint: {len(missing)} trainable tensors missing.')
        result = model.load_state_dict(checkpoint['model'], strict=False)
        if result.unexpected_keys:
            raise ModelUnavailable(f'Unexpected adapter tensors: {result.unexpected_keys[:3]}')
        del checkpoint
        model = model.eval().to(device=device, dtype=torch.bfloat16)
        model.get_prompt(drop_audio=True, drop_video=False, device=device, set_prompt=True)
        # Compile ONLY the released inference functions. Importing the full CLI would
        # execute Hydra/Fairseq training registration and researcher-local paths.
        path = self.runtime / 'decode_scripts/infer_s2s_beamsearch_s3_chime_llm.py'
        from .decoding import build_search
        self._search = build_search(path, LOGGER)
        from .preprocessing import FacePreprocessor
        from .visual_speech import VisualSpeechDetector
        try:
            visual_speech = VisualSpeechDetector(device=device, checkpoint='talkset',
                                                weights_dir=self.assets / 'visual_speech')
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise ModelUnavailable(f'The visual speech filter could not load: {exc}') from exc
        self.preprocessor = FacePreprocessor(self.assets / '20words_mean_face.npy', device=device, mean=float(full_cfg.task.get('image_mean', .421)), std=float(full_cfg.task.get('image_std', .165)), crop_size=int(full_cfg.task.get('image_crop_size', 88)), visual_speech=visual_speech)
        self.model = model

    def prepare_frame(self, frame, *, cache, speech_gate):
        """Called by the same session worker, only between scheduled updates."""
        with self._lock:
            self.preprocessor.prepare_frame(frame, cache, speech_gate=speech_gate)

    def inspect_activity(self, frames, fps=25, *, cache=None, speech_gate=True) -> dict:
        """Inspect a camera window without uploading or decoding FLARE inputs.

        The caller supplies the session's activity cache and exact sample times.
        Stop preprocessing after the activity/quality checks; recognition alone
        needs aligned FLARE images. No hypotheses are generated here.
        """
        if self.model is None:
            raise ModelUnavailable(self.status()['detail'])
        with self._lock:
            start = time.perf_counter()
            prepared = self.preprocessor(frames, fps, cache=cache, speech_gate=speech_gate,
                                         activity_only=True)
            elapsed_ms = (time.perf_counter() - start) * 1000
            activity = prepared.activity or {'state': 'bypassed', 'suppress': False,
                                            'detail': 'Speech filter off.', 'metrics': {}}
            return {'activity': activity, 'quality': prepared.quality,
                    'detail': activity['detail'], 'model': self.name,
                    'latency_ms': round(elapsed_ms),
                    'timings_ms': {'preprocessing': round(elapsed_ms, 1),
                                   'upload': 0.0, 'inference': 0.0}}

    def decode_utterance(self, frames, fps=25, *, cache=None, verify_visual=True) -> dict:
        """Decode the full retained utterance after the caller admits speech.

        Live drafts and final refinement use identical recognition code.
        The session controller, not this method, decides when detected speech
        warrants a decode. Use a separate utterance preprocessing cache so this
        pass cannot advance the activity detector's evidence watermark.
        """
        return self.transcribe(frames, fps, cache=cache, speech_gate=False, verify_visual=verify_visual)

    # Explicit name for callers performing the one final pass at an utterance end.
    refine_utterance = decode_utterance

    def transcribe(self, frames, fps=25, *, cache=None, speech_gate=True, verify_visual=False) -> dict:
        if self.model is None:
            raise ModelUnavailable(self.status()['detail'])
        # Serialise one GPU model; the HTTP layer bounds admission and clip lengths.
        with self._lock:
            import torch
            start = time.perf_counter()
            prepared = self.preprocessor(frames, fps, cache=cache, speech_gate=speech_gate)
            prepared_at = time.perf_counter()
            activity = prepared.activity or {'state': 'bypassed', 'suppress': False,
                                            'detail': 'Still-mouth filter off.', 'metrics': {}}
            if activity['suppress']:
                return {'text': '', 'latency_ms': round((prepared_at - start) * 1000),
                        'timings_ms': {'preprocessing': round((prepared_at - start) * 1000, 1),
                                       'upload': 0.0, 'inference': 0.0},
                        'activity': activity, 'detail': activity['detail'],
                        'quality': prepared.quality, 'model': self.name}
            tensors = [torch.from_numpy(array).to(device='cuda', dtype=torch.bfloat16) for array in (prepared.face, prepared.lips)]
            frame_count = tensors[0].shape[2]
            sample = {'net_input': {'source': {'audio': None, 'video': tuple(tensors), 'face_lip_mask': (torch.ones(1, device='cuda'), torch.ones(1, device='cuda'))}, 'padding_mask': torch.zeros((1, frame_count), dtype=torch.bool, device='cuda')}}
            uploaded_at = time.perf_counter()
            recognition = None
            with torch.inference_mode():
                if verify_visual:
                    from .grounding import capture_candidate, verify_candidate
                    with capture_candidate(self._search) as captured:
                        hypotheses, _ = self._search([self.model], sample, beam_size=self.beam,
                                                     audio_only=False, video_only=True, temp=1.2)
                    searched_at = time.perf_counter()
                    if len(captured) != 1:
                        raise RuntimeError('Visual verification did not receive the decoder evidence.')
                    recognition = verify_candidate(self.model, sample, captured[0])
                    replacement = recognition.pop('_replacement_tokens', None)
                    if replacement is not None:
                        hypotheses = [self.model.llm_tokenizer.decode(replacement.tolist(), skip_special_tokens=True)]
                    original_repetition = recognition.get('original_repetition', recognition.get('repetition', {}))
                    # Counts/scores only: distinguish model loops from duplicate
                    # display updates without recording words or camera frames.
                    LOGGER.info('Recognition accepted=%s state=%s repeated_runs=%s alternatives=%s',
                                recognition['accepted'], recognition.get('state'),
                                original_repetition.get('runs', []),
                                recognition.get('alternative_candidates_scored', 0))
                else:
                    hypotheses, _ = self._search([self.model], sample, beam_size=self.beam,
                                                 audio_only=False, video_only=True, temp=1.2)
                    searched_at = time.perf_counter()
            decoded_at = time.perf_counter()
            text = ' '.join(hypotheses[0].replace('<sil>', '').split())
            if recognition is not None and not recognition['accepted']:
                text = ''
            return {'text': text, 'latency_ms': round((decoded_at - start) * 1000),
                    'timings_ms': {'preprocessing': round((prepared_at - start) * 1000, 1),
                                   'upload': round((uploaded_at - prepared_at) * 1000, 1),
                                   'inference': round((searched_at - uploaded_at) * 1000, 1),
                                   'grounding': round((decoded_at - searched_at) * 1000, 1)},
                    'recognition': recognition,
                    'activity': activity, 'quality': prepared.quality, 'model': self.name}

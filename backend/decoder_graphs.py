"""Experimental exact-shape Qwen CUDA graphs, used only by the benchmark.

No padding or cache approximation. Shapes not already captured pay setup costs;
this is deliberately not enabled in the interactive server.
"""
from collections import OrderedDict


class DecoderGraphs:
    def __init__(self, forward, max_entries=128):
        self.forward = forward
        self.entries = OrderedDict()
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def __call__(self, *args, **kwargs):
        import torch
        ids = kwargs.get('input_ids')
        if args or ids is None or ids.shape[1] != 1 or not kwargs.get('past_key_values'):
            return self.forward(*args, **kwargs)

        def signature(value):
            if isinstance(value, torch.Tensor):
                return (tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
            if isinstance(value, (tuple, list)):
                return tuple(signature(item) for item in value)
            return value

        def clone(value):
            if isinstance(value, torch.Tensor):
                return value.clone()
            if isinstance(value, (tuple, list)):
                return tuple(clone(item) for item in value)
            return value

        def update(target, value):
            if isinstance(value, torch.Tensor):
                target.copy_(value)
            elif isinstance(value, (tuple, list)):
                for destination, source in zip(target, value):
                    update(destination, source)

        key = tuple((name, signature(value)) for name, value in sorted(kwargs.items()))
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            static = {name: clone(value) for name, value in kwargs.items()}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.forward(**static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.forward(**static)
            entry = static, graph, output
            self.entries[key] = entry
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
        else:
            self.hits += 1
            self.entries.move_to_end(key)
            for name, value in kwargs.items():
                update(entry[0][name], value)
        entry[1].replay()
        return entry[2]

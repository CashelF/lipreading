import { useEffect } from 'react';
import type { RefObject } from 'react';

// Modal focus: Escape closes (unless the close button is disabled) and Tab
// stays inside the dialog even while recording disables most controls.
export function useDialogFocus(dialogRef: RefObject<HTMLElement | null>, closeRef: RefObject<HTMLButtonElement | null>, onClose: () => void) {
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    const listener = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && closeRef.current && !closeRef.current.disabled) onClose();
      if (event.key !== 'Tab') return;
      const controls = dialogRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), [tabindex="0"]');
      const first = controls?.[0];
      const last = controls?.[controls.length - 1];
      if (!first || !last) { event.preventDefault(); dialogRef.current?.focus(); return; }
      if (!Array.from(controls ?? []).includes(document.activeElement as HTMLElement)) {
        event.preventDefault(); (event.shiftKey ? last : first).focus(); return;
      }
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', listener);
    return () => { document.removeEventListener('keydown', listener); previous?.focus(); };
  }, [dialogRef, closeRef, onClose]);
}

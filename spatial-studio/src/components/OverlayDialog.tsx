import { X } from 'lucide-react'
import { useEffect } from 'react'

interface OverlayDialogProps {
  open: boolean
  title: string
  children: React.ReactNode
  onClose: () => void
}

export function OverlayDialog({ open, title, children, onClose }: OverlayDialogProps) {
  useEffect(() => {
    if (!open) return undefined
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose, open])

  if (!open) return null
  return (
    <div className="dialog-backdrop" role="presentation" onPointerDown={onClose}>
      <section
        className="dialog-card overlay-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="overlay-dialog-title"
        onPointerDown={(event) => event.stopPropagation()}
      >
        <header>
          <h2 id="overlay-dialog-title">{title}</h2>
          <button type="button" aria-label={`关闭${title}`} onClick={onClose} autoFocus>
            <X aria-hidden="true" />
          </button>
        </header>
        <div className="overlay-dialog-body">{children}</div>
      </section>
    </div>
  )
}

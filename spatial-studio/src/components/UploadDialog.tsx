import { FileImage, Upload, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

interface UploadDialogProps {
  open: boolean
  maxUploadMb: number
  onClose: () => void
  onFile: (file: File) => void
}

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/bmp', 'image/gif', 'image/tiff']

export function UploadDialog({ open, maxUploadMb, onClose, onFile }: UploadDialogProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return undefined
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose, open])

  if (!open) return null

  function accept(file: File | undefined) {
    if (!file) return
    if (file.size > maxUploadMb * 1024 * 1024) {
      setError(`图片不能超过 ${maxUploadMb} MB`)
      return
    }
    const extensionAccepted = /\.(png|jpe?g|webp|bmp|gif|tiff?)$/i.test(file.name)
    if (file.type && !ACCEPTED_TYPES.includes(file.type) && !extensionAccepted) {
      setError('请选择 PNG、JPEG、WebP、BMP、GIF 或 TIFF 图片')
      return
    }
    setError(null)
    onFile(file)
  }

  return (
    <div className="dialog-backdrop" role="presentation" onPointerDown={onClose}>
      <section
        className="dialog-card upload-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="upload-title"
        onPointerDown={(event) => event.stopPropagation()}
      >
        <header>
          <div>
            <h2 id="upload-title">上传平面图</h2>
            <p>推荐使用干净的俯视图、楼层图或户型图。</p>
          </div>
          <button type="button" aria-label="关闭上传窗口" onClick={onClose} autoFocus>
            <X aria-hidden="true" />
          </button>
        </header>
        <button
          className={dragging ? 'drop-zone dragging' : 'drop-zone'}
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragEnter={(event) => {
            event.preventDefault()
            setDragging(true)
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault()
            setDragging(false)
            accept(event.dataTransfer.files[0])
          }}
        >
          <span className="upload-icon"><Upload aria-hidden="true" /></span>
          <strong>拖入图片，或点击选择</strong>
          <span>最大 {maxUploadMb} MB；服务器会重新编码以保证安全</span>
        </button>
        <input
          ref={inputRef}
          className="sr-only"
          type="file"
          accept="image/png,image/jpeg,image/webp,image/bmp,image/gif,image/tiff,.tif,.tiff"
          onChange={(event) => accept(event.target.files?.[0])}
        />
        {error ? <p className="dialog-error" role="alert">{error}</p> : null}
        <div className="upload-guidance">
          <FileImage aria-hidden="true" />
          <p><strong>实景照片说明</strong><span>单张透视照片默认不会被当成完整导航地图；建议先生成俯视占用图。</span></p>
        </div>
      </section>
    </div>
  )
}

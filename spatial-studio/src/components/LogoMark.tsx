interface LogoMarkProps {
  size?: number
}

export function LogoMark({ size = 38 }: LogoMarkProps) {
  return (
    <svg
      aria-hidden="true"
      className="logo-mark"
      height={size}
      viewBox="0 0 48 48"
      width={size}
    >
      <path
        d="M24 3 42 13.5v21L24 45 6 34.5v-21L24 3Z"
        fill="none"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="3"
      />
      <path
        d="m13 17 11 6.5L35 17M13 31l11-6.5L35 31"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="3"
      />
      <circle cx="13" cy="17" fill="currentColor" r="2.5" />
      <circle cx="35" cy="17" fill="currentColor" r="2.5" />
      <circle cx="13" cy="31" fill="currentColor" r="2.5" />
      <circle cx="35" cy="31" fill="currentColor" r="2.5" />
    </svg>
  )
}

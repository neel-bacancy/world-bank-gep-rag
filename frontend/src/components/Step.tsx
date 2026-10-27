import type { ReactNode } from 'react'

/** One stage of a pipeline: a number, what the stage does, its timing if measured, and its output. */
export function Step({
  n,
  title,
  what,
  timing,
  children,
}: {
  n: number
  title: string
  what: string
  timing?: number
  children: ReactNode
}) {
  return (
    <section className="step">
      <div className="step-marker">{n}</div>
      <div className="step-main">
        <header className="step-head">
          <div>
            <h3>{title}</h3>
            <p className="step-what">{what}</p>
          </div>
          {timing !== undefined && <span className="timing">{timing} ms</span>}
        </header>
        <div className="step-body">{children}</div>
      </div>
    </section>
  )
}

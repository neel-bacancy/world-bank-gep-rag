/** Renders the answer's paragraphs. Citations like [S1] or [S1, S2] become buttons that jump to that source. */
export function AnswerText({ text, onCite }: { text: string; onCite: (n: number) => void }) {
  const lines = text.split('\n').filter((line) => line.trim())
  return (
    <div className="answer-text">
      {lines.map((line, i) => (
        <p key={i}>{renderLine(line.replace(/^\s*[-*]\s+/, '• '), onCite)}</p>
      ))}
    </div>
  )
}

function renderLine(line: string, onCite: (n: number) => void) {
  return line.split(/(\[S\d+(?:,\s*S\d+)*\])/g).map((part, i) => {
    if (!/^\[S\d/.test(part)) return part
    return (
      <span key={i} className="cite-group">
        {(part.match(/\d+/g) ?? []).map((n) => (
          <button key={n} className="cite" onClick={() => onCite(Number(n))}>
            S{n}
          </button>
        ))}
      </span>
    )
  })
}

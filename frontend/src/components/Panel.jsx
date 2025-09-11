// src/components/Panel.jsx
export default function Panel({ title, children }) {
    return (
      <section className="panel">
        <header className="panel-header">
          <h2>{title}</h2>
        </header>
        <div className="panel-body">{children}</div>
      </section>
    );
  }
  
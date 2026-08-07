/**
 * The onboarding form.
 *
 * Replaces the "go open an app while I watch" step. Typing "Google Docs" or
 * pasting a link takes two seconds and is exactly right; watching the frontmost
 * window for twenty seconds was slow, guessed wrong when anything else came to
 * the front, and couldn't capture a specific URL at all.
 */

import { useState } from 'react';

export interface FormField {
  key: string;
  label: string;
  hint?: string;
  placeholder?: string;
}

interface Props {
  title: string;
  intro: string;
  fields: FormField[];
  onSubmit: (values: Record<string, string>) => void;
  onSkip: () => void;
}

export default function OnboardingForm({
  title,
  intro,
  fields,
  onSubmit,
  onSkip,
}: Props) {
  const [values, setValues] = useState<Record<string, string>>({});

  const filled = Object.values(values).filter((v) => v.trim()).length;

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    onSubmit(values);
  };

  return (
    <form className="onboard" onSubmit={submit}>
      <h2>{title}</h2>
      <p className="onboard-intro">{intro}</p>

      <div className="onboard-fields">
        {fields.map((field) => (
          <label key={field.key} className="onboard-field">
            <span className="onboard-label">{field.label}</span>
            {field.hint && <span className="onboard-hint">{field.hint}</span>}
            <input
              value={values[field.key] ?? ''}
              placeholder={field.placeholder}
              spellCheck={false}
              autoComplete="off"
              onChange={(e) =>
                setValues((current) => ({
                  ...current,
                  [field.key]: e.target.value,
                }))
              }
            />
          </label>
        ))}
      </div>

      <div className="onboard-actions">
        <button type="submit" className="yes">
          {filled ? `Save ${filled}` : 'Save'}
        </button>
        <button type="button" onClick={onSkip}>
          Skip all
        </button>
      </div>
      <p className="onboard-note">
        Leave anything blank and it just gets skipped. You can redo this later
        with <code>jarvis onboard --restart</code>.
      </p>
    </form>
  );
}

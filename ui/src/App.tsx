/**
 * The screen. Open http://localhost:5173 in a Chrome tab and leave it there.
 *
 * gsap tweens the glob between state presets rather than snapping to them —
 * that's the "smoothly, not suddenly" part of your spec. Idle -> listening is
 * the slowest transition on purpose, since that's the one you actually watch.
 */

import { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import AbstractBall from './components/AbstractBall';
import OnboardingForm from './components/OnboardingForm';
import { useJarvis } from './hooks/useJarvis';
import { STATES, TRANSITION, morphForLevel, type GlobConfig } from './lib/globStates';

const LABEL: Record<string, string> = {
  idle: 'listening for "hey Jarvis"',
  listening: 'go ahead',
  thinking: 'thinking',
  speaking: 'speaking',
  offline: 'daemon not running',
};

export default function App() {
  const jarvis = useJarvis();
  const [config, setConfig] = useState<GlobConfig>(STATES.offline);
  const tweened = useRef<GlobConfig>({ ...STATES.offline });
  const [typed, setTyped] = useState('');

  // Tween every numeric uniform toward the new state's preset. gsap writes into
  // `tweened.current`, and onUpdate publishes a snapshot so the glob re-renders.
  useEffect(() => {
    const target = STATES[jarvis.state];
    const duration = TRANSITION[jarvis.state];

    const tween = gsap.to(tweened.current, {
      duration,
      ease: 'power2.inOut',
      perlinTime: target.perlinTime,
      perlinMorph: target.perlinMorph,
      perlinDNoise: target.perlinDNoise,
      chromaRGBr: target.chromaRGBr,
      chromaRGBg: target.chromaRGBg,
      chromaRGBb: target.chromaRGBb,
      chromaRGBn: target.chromaRGBn,
      chromaRGBm: target.chromaRGBm,
      tintMix: target.tintMix,
      gain: target.gain,
      cameraSpeedY: target.cameraSpeedY,
      cameraZoom: target.cameraZoom,
      onUpdate: () => setConfig({ ...tweened.current }),
      onComplete: () => setConfig({ ...tweened.current }),
    });

    // Colour is a vec3, so it gets its own tween writing into the array.
    const colour = gsap.to(tweened.current.tint, {
      duration,
      ease: 'power2.inOut',
      0: target.tint[0],
      1: target.tint[1],
      2: target.tint[2],
    });

    return () => {
      tween.kill();
      colour.kill();
    };
  }, [jarvis.state]);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!typed.trim()) return;
    jarvis.sendText(typed);
    setTyped('');
  };

  return (
    <div className="app">
      <div className="stage">
        <AbstractBall
          {...config}
          perlinMorph={morphForLevel(jarvis.state, jarvis.level)}
          className="glob"
        />
      </div>

      <div className="readout">
        <div className={`status status-${jarvis.state}`}>
          {jarvis.connected ? LABEL[jarvis.state] : LABEL.offline}
        </div>

        {jarvis.transcript && <p className="transcript">{jarvis.transcript}</p>}
      </div>

      {jarvis.onboardForm && (
        <OnboardingForm
          title={jarvis.onboardForm.title}
          intro={jarvis.onboardForm.intro}
          fields={jarvis.onboardForm.fields}
          onSubmit={jarvis.submitOnboarding}
          onSkip={() => jarvis.submitOnboarding({})}
        />
      )}

      {jarvis.capability && (
        <div className="permission" role="dialog" aria-live="assertive">
          <p className="permission-prompt">{jarvis.capability.prompt}</p>
          <p className="permission-note">
            Saying yes allows <code>{jarvis.capability.capability}</code> from now
            on — you won't be asked again.
          </p>
          <div className="permission-actions">
            <button className="yes" onClick={() => jarvis.answerCapability(true)}>
              Allow
            </button>
            <button className="no" onClick={() => jarvis.answerCapability(false)}>
              Not now
            </button>
          </div>
        </div>
      )}

      <form className="composer" onSubmit={submit}>
        <input
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder="or type instead…"
          spellCheck={false}
        />
        <button
          type="button"
          className="mic"
          onClick={jarvis.state === 'speaking' ? jarvis.cancel : jarvis.triggerListen}
        >
          {jarvis.state === 'speaking' ? 'stop' : 'talk'}
        </button>
      </form>

      {!jarvis.connected && (
        <p className="hint">
          Start the daemon: <code>cd core &amp;&amp; python -m jarvis run</code>
        </p>
      )}
    </div>
  );
}

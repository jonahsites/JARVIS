/**
 * Connection to the daemon. This replaces useWebRTCAudioSession.
 *
 * The important difference: no audio happens in the browser. The microphone,
 * wake word, transcription and speech all live in the Python process, so the
 * loop keeps working whether or not this tab is open — which is what you want
 * from something you shout at from across the room.
 *
 * This hook only listens. It receives state and audio level and renders them.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormField } from '../components/OnboardingForm';
import type { JarvisState } from '../lib/globStates';

const BUS_URL = `ws://127.0.0.1:${import.meta.env.VITE_BUS_PORT ?? 8765}`;

export interface OnboardForm {
  id: string;
  title: string;
  intro: string;
  fields: FormField[];
}

export interface CapabilityPrompt {
  id: string;
  capability: string;
  prompt: string;
}

export interface JarvisSession {
  connected: boolean;
  state: JarvisState;
  level: number;
  transcript: string;
  spoken: string;
  capability: CapabilityPrompt | null;
  onboardForm: OnboardForm | null;
  answerCapability: (granted: boolean) => void;
  submitOnboarding: (values: Record<string, string>) => void;
  triggerListen: () => void;
  cancel: () => void;
  sendText: (text: string) => void;
}

export function useJarvis(): JarvisSession {
  const socket = useRef<WebSocket | null>(null);
  const retry = useRef<number>(0);
  const decay = useRef<number>(0);

  const [connected, setConnected] = useState(false);
  const [state, setState] = useState<JarvisState>('offline');
  const [level, setLevel] = useState(0);
  const [transcript, setTranscript] = useState('');
  const [spoken, setSpoken] = useState('');
  const [capability, setCapability] = useState<CapabilityPrompt | null>(null);
  const [onboardForm, setOnboardForm] = useState<OnboardForm | null>(null);

  const send = useCallback((type: string, payload: Record<string, unknown> = {}) => {
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify({ type, payload }));
    }
  }, []);

  useEffect(() => {
    let closed = false;

    const connect = () => {
      if (closed) return;
      const ws = new WebSocket(BUS_URL);
      socket.current = ws;

      ws.onopen = () => {
        setConnected(true);
        retry.current = 0;
      };

      ws.onclose = () => {
        setConnected(false);
        setState('offline');
        if (closed) return;
        // Back off to 5s so a daemon that's down doesn't spam the console.
        retry.current = Math.min(retry.current + 1, 10);
        window.setTimeout(connect, Math.min(500 * retry.current, 5000));
      };

      ws.onerror = () => ws.close();

      ws.onmessage = (event) => {
        const { type, payload } = JSON.parse(event.data);
        switch (type) {
          case 'state':
            setState(payload.state as JarvisState);
            if (payload.state === 'idle') setTranscript('');
            break;
          case 'level':
            setLevel(payload.rms as number);
            break;
          case 'transcript':
            setTranscript(payload.text as string);
            break;
          case 'speech':
            setSpoken(payload.text as string);
            break;
          case 'capability':
            setCapability({
              id: payload.id,
              capability: payload.capability,
              prompt: payload.prompt,
            });
            break;
          case 'onboard_form':
            setOnboardForm({
              id: payload.id,
              title: payload.title,
              intro: payload.intro,
              fields: payload.fields as FormField[],
            });
            break;
        }
      };
    };

    connect();
    return () => {
      closed = true;
      socket.current?.close();
    };
  }, []);

  // Levels arrive in bursts; without decay the glob would freeze mid-morph
  // whenever the daemon went quiet between frames.
  useEffect(() => {
    const timer = window.setInterval(() => {
      setLevel((current) => {
        const next = current * 0.82;
        return next < 0.005 ? 0 : next;
      });
    }, 60);
    decay.current = timer;
    return () => window.clearInterval(timer);
  }, []);

  const answerCapability = useCallback(
    (granted: boolean) => {
      if (!capability) return;
      send('capability_reply', { id: capability.id, granted });
      setCapability(null);
    },
    [capability, send],
  );

  const submitOnboarding = useCallback(
    (values: Record<string, string>) => {
      if (!onboardForm) return;
      send('onboard_submit', { id: onboardForm.id, values });
      setOnboardForm(null);
    },
    [onboardForm, send],
  );

  return {
    connected,
    state,
    level,
    transcript,
    spoken,
    capability,
    onboardForm,
    answerCapability,
    submitOnboarding,
    triggerListen: useCallback(() => send('hotkey'), [send]),
    cancel: useCallback(() => send('cancel'), [send]),
    sendText: useCallback((text: string) => send('text_input', { text }), [send]),
  };
}

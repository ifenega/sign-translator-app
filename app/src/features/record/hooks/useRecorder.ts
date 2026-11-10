import type { CameraView } from 'expo-camera';
import { useEffect, useRef, useState } from 'react';
import { Alert } from 'react-native';

import type { RecordedVideo } from '../types';

export const COUNTDOWN_SECONDS = 3;
export const RECORD_SECONDS = 3;

export type RecorderPhase =
  | { name: 'ready' }
  | { name: 'countdown'; secondsLeft: number }
  | { name: 'recording' };

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Runs the 3-2-1 countdown and the 3-second recording, then hands the video over.
export function useRecorder(onRecorded: (video: RecordedVideo) => void) {
  const cameraRef = useRef<CameraView>(null);
  const [phase, setPhase] = useState<RecorderPhase>({ name: 'ready' });
  // Set when the recorder closes, so a countdown or recording that is still
  // running stops quietly instead of handing over a video nobody wants.
  const closed = useRef(false);

  useEffect(() => {
    closed.current = false;
    return () => {
      closed.current = true;
    };
  }, []);

  async function start() {
    for (let secondsLeft = COUNTDOWN_SECONDS; secondsLeft > 0; secondsLeft--) {
      setPhase({ name: 'countdown', secondsLeft });
      await wait(1000);
      if (closed.current) return;
    }

    setPhase({ name: 'recording' });
    try {
      const result = await cameraRef.current?.recordAsync({ maxDuration: RECORD_SECONDS });
      if (closed.current) return;
      setPhase({ name: 'ready' });
      if (result) {
        onRecorded({ uri: result.uri });
      }
    } catch {
      if (closed.current) return;
      setPhase({ name: 'ready' });
      Alert.alert('Recording failed', 'Something went wrong with the camera. Please try again.');
    }
  }

  return { cameraRef, phase, start };
}

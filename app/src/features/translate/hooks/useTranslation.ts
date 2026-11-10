import * as Haptics from 'expo-haptics';
import { useRef, useState } from 'react';

import { translateVideo } from '../api/translateVideo';
import type { PickedVideo, TranslationState } from '../types';

export function useTranslation() {
  const [state, setState] = useState<TranslationState>({ status: 'idle' });
  // Numbers each request, so an answer that arrives after the user has
  // cancelled (or picked another video) is ignored.
  const latestRequest = useRef(0);

  async function translate(video: PickedVideo) {
    const request = ++latestRequest.current;
    setState({ status: 'translating', video });
    try {
      const result = await translateVideo(video);
      console.log(
        `Translation of "${video.name}": "${result.text}" (${Math.round(result.confidence * 100)}% confident)`,
      );
      if (request === latestRequest.current) {
        setState({ status: 'done', video, text: result.text });
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }
    } catch (error) {
      console.warn(`Translation of "${video.name}" failed:`, error);
      if (request === latestRequest.current) {
        setState({ status: 'failed', video });
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
      }
    }
  }

  function clear() {
    latestRequest.current++;
    setState({ status: 'idle' });
  }

  return { state, translate, clear };
}

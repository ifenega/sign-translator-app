import type { PickedVideo, TranslationResult } from '../types';

// Placeholder until the server's POST /predict endpoint exists. It waits a
// moment and returns a fixed answer so the screens can be built and tried.
// Replace the body with an upload of `video.uri` to the server.
export async function translateVideo(video: PickedVideo): Promise<TranslationResult> {
  await new Promise((resolve) => setTimeout(resolve, 2500));
  return { text: 'Hello', confidence: 0.86 };
}

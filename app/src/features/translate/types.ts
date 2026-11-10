export type PickedVideo = {
  uri: string;
  name: string;
};

export type TranslationResult = {
  text: string;
  confidence: number; // 0 to 1: how sure the model is
};

export type TranslationState =
  | { status: 'idle' }
  | { status: 'translating'; video: PickedVideo }
  | { status: 'done'; video: PickedVideo; text: string }
  | { status: 'failed'; video: PickedVideo };

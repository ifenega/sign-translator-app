import { Modal } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import type { RecordedVideo } from '../types';
import { Recorder } from './Recorder';

type Props = {
  visible: boolean;
  onRecorded: (video: RecordedVideo) => void;
  onClose: () => void;
};

// Opens the camera over the current screen. The camera only exists while the
// modal is open, so it is switched off as soon as it closes.
export function RecordModal({ visible, onRecorded, onClose }: Props) {
  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="fullScreen"
      onRequestClose={onClose}
    >
      {/* A modal sits outside the app's safe-area setup, so it needs its own. */}
      <SafeAreaProvider>
        {visible && <Recorder onRecorded={onRecorded} onClose={onClose} />}
      </SafeAreaProvider>
    </Modal>
  );
}

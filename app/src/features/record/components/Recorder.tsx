import { CameraView, useCameraPermissions } from 'expo-camera';
import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { colors, fonts } from '@/shared/theme';

import { RECORD_SECONDS, useRecorder } from '../hooks/useRecorder';
import type { RecordedVideo } from '../types';
import { CameraPermissionMessage } from './CameraPermissionMessage';
import { CloseButton } from './CloseButton';
import { FrameGuide } from './FrameGuide';
import { RecordButton } from './RecordButton';
import { RecordingBar } from './RecordingBar';

type Props = {
  onRecorded: (video: RecordedVideo) => void;
  onClose: () => void;
};

// The full-screen camera: line up with the outline, press record, a 3-2-1
// countdown, then 3 seconds of recording.
export function Recorder({ onRecorded, onClose }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [isCameraReady, setIsCameraReady] = useState(false);
  const { cameraRef, phase, start } = useRecorder(onRecorded);

  if (!permission) {
    // Still checking whether the camera is allowed.
    return <View style={styles.screen} />;
  }

  if (!permission.granted) {
    return (
      <CameraPermissionMessage
        canAskAgain={permission.canAskAgain}
        onRequest={requestPermission}
        onClose={onClose}
      />
    );
  }

  return (
    <View style={styles.screen}>
      <StatusBar style="light" />
      {/* Muted: signing needs no sound, and it means no microphone permission. */}
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        facing="front"
        mode="video"
        mute
        onCameraReady={() => setIsCameraReady(true)}
      />

      <SafeAreaView style={styles.overlay} edges={['top', 'bottom']}>
        <View style={styles.topBar}>
          <CloseButton onPress={onClose} />
        </View>

        <View style={styles.middle}>
          <FrameGuide />
          {phase.name === 'countdown' && (
            <Text style={styles.countdown} accessibilityLiveRegion="assertive">
              {phase.secondsLeft}
            </Text>
          )}
        </View>

        <View style={styles.bottom}>
          {phase.name === 'ready' && (
            <>
              <Text style={styles.hint}>Fit your head and shoulders in the outline</Text>
              <RecordButton disabled={!isCameraReady} onPress={start} />
            </>
          )}
          {phase.name === 'recording' && <RecordingBar seconds={RECORD_SECONDS} />}
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.text,
  },
  overlay: {
    flex: 1,
  },
  topBar: {
    padding: 16,
    alignItems: 'flex-start',
  },
  middle: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  countdown: {
    position: 'absolute',
    fontFamily: fonts.bold,
    fontSize: 120,
    color: colors.white,
    textShadowColor: 'rgba(0,0,0,0.4)',
    textShadowRadius: 12,
  },
  bottom: {
    minHeight: 150,
    paddingHorizontal: 32,
    paddingBottom: 24,
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: 20,
  },
  hint: {
    fontFamily: fonts.semibold,
    fontSize: 15,
    color: colors.white,
    textAlign: 'center',
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowRadius: 6,
  },
});

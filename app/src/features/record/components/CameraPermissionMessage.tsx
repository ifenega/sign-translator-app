import { Linking, Pressable, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { colors, fonts } from '@/shared/theme';

import { CloseButton } from './CloseButton';

type Props = {
  canAskAgain: boolean;
  onRequest: () => void;
  onClose: () => void;
};

// Shown when the app may not use the camera yet. If the phone won't show the
// permission prompt again, the button opens Settings instead.
export function CameraPermissionMessage({ canAskAgain, onRequest, onClose }: Props) {
  return (
    <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
      <View style={styles.topBar}>
        <CloseButton onPress={onClose} />
      </View>
      <View style={styles.body}>
        <Text style={styles.title}>Allow the camera</Text>
        <Text style={styles.text}>Sign Translator needs the camera to record you signing.</Text>
        <Pressable
          onPress={canAskAgain ? onRequest : () => Linking.openSettings()}
          accessibilityRole="button"
          style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
        >
          <Text style={styles.buttonText}>{canAskAgain ? 'Allow camera' : 'Open Settings'}</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.text,
  },
  topBar: {
    padding: 16,
  },
  body: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 32,
    gap: 12,
  },
  title: {
    fontFamily: fonts.bold,
    fontSize: 24,
    color: colors.white,
    textAlign: 'center',
  },
  text: {
    fontFamily: fonts.regular,
    fontSize: 16,
    lineHeight: 23,
    color: 'rgba(255,255,255,0.75)',
    textAlign: 'center',
  },
  button: {
    marginTop: 12,
    height: 54,
    borderRadius: 27,
    backgroundColor: colors.white,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonPressed: {
    opacity: 0.85,
  },
  buttonText: {
    fontFamily: fonts.semibold,
    fontSize: 16,
    color: colors.text,
  },
});

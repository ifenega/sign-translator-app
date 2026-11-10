import Feather from '@expo/vector-icons/Feather';
import { Animated, Pressable, StyleSheet, Text, View } from 'react-native';

import { colors, fonts, softCycle } from '@/shared/theme';

import { cycleColor, useColorCycle } from '../hooks/useColorCycle';
import { Card } from './Card';

type Props = {
  onRecordVideo: () => void;
  onChooseVideo: () => void;
};

export function UploadCard({ onRecordVideo, onChooseVideo }: Props) {
  const progress = useColorCycle(softCycle.length, 10000);

  return (
    <Card style={styles.card}>
      <View style={styles.panel}>
        <Animated.View style={[styles.circle, { backgroundColor: cycleColor(progress, softCycle) }]}>
          <Feather name="video" size={32} color={colors.text} />
        </Animated.View>
      </View>

      <View style={styles.text}>
        <Text style={styles.title}>Record or upload a video</Text>
        <Text style={styles.subtitle}>Keep your hands in view</Text>
      </View>

      <View style={styles.buttons}>
        <Pressable
          onPress={onRecordVideo}
          accessibilityRole="button"
          style={({ pressed }) => [styles.button, styles.primaryButton, pressed && styles.buttonPressed]}
        >
          <Feather name="video" size={19} color={colors.white} />
          <Text style={[styles.buttonText, styles.primaryButtonText]}>Record video</Text>
        </Pressable>
        <Pressable
          onPress={onChooseVideo}
          accessibilityRole="button"
          style={({ pressed }) => [styles.button, styles.secondaryButton, pressed && styles.buttonPressed]}
        >
          <Feather name="upload" size={19} color={colors.text} />
          <Text style={styles.buttonText}>Choose video</Text>
        </Pressable>
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: 10,
    paddingBottom: 20,
    gap: 18,
  },
  panel: {
    height: 180,
    borderRadius: 20,
    backgroundColor: colors.panel,
    alignItems: 'center',
    justifyContent: 'center',
  },
  circle: {
    width: 84,
    height: 84,
    borderRadius: 42,
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 0 0 10px rgba(255,255,255,0.7)',
  },
  text: {
    alignItems: 'center',
    gap: 4,
  },
  title: {
    fontFamily: fonts.semibold,
    fontSize: 20,
    color: colors.text,
  },
  subtitle: {
    fontFamily: fonts.regular,
    fontSize: 15,
    color: colors.textMuted,
  },
  buttons: {
    marginHorizontal: 10,
    gap: 10,
  },
  button: {
    height: 54,
    borderRadius: 27,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
  },
  primaryButton: {
    backgroundColor: colors.text,
  },
  secondaryButton: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
  },
  buttonPressed: {
    opacity: 0.85,
  },
  buttonText: {
    fontFamily: fonts.semibold,
    fontSize: 16,
    color: colors.text,
  },
  primaryButtonText: {
    color: colors.white,
  },
});

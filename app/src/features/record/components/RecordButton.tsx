import { Pressable, StyleSheet, View } from 'react-native';

import { colors } from '@/shared/theme';

type Props = {
  disabled: boolean;
  onPress: () => void;
};

export function RecordButton({ disabled, onPress }: Props) {
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel="Start recording"
      style={({ pressed }) => [styles.ring, disabled && styles.disabled, pressed && styles.pressed]}
    >
      <View style={styles.dot} />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  ring: {
    width: 78,
    height: 78,
    borderRadius: 39,
    borderWidth: 4,
    borderColor: colors.white,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dot: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: colors.record,
  },
  disabled: {
    opacity: 0.4,
  },
  pressed: {
    transform: [{ scale: 0.94 }],
  },
});

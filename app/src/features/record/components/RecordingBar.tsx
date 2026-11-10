import { useEffect, useState } from 'react';
import { Animated, Easing, StyleSheet, Text, View } from 'react-native';

import { colors, fonts } from '@/shared/theme';

type Props = {
  seconds: number;
};

// Shown while recording: a "Recording" label and a bar that fills up over the
// length of the recording.
export function RecordingBar({ seconds }: Props) {
  const [progress] = useState(() => new Animated.Value(0));

  useEffect(() => {
    Animated.timing(progress, {
      toValue: 1,
      duration: seconds * 1000,
      easing: Easing.linear,
      useNativeDriver: false, // width can't use the native driver
    }).start();
  }, [progress, seconds]);

  const width = progress.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] });

  return (
    <View style={styles.container} accessibilityLiveRegion="polite">
      <View style={styles.label}>
        <View style={styles.redDot} />
        <Text style={styles.labelText}>Recording</Text>
      </View>
      <View style={styles.track}>
        <Animated.View style={[styles.fill, { width }]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    alignItems: 'center',
    gap: 14,
  },
  label: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  redDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.record,
  },
  labelText: {
    fontFamily: fonts.semibold,
    fontSize: 15,
    color: colors.white,
  },
  track: {
    width: '100%',
    height: 6,
    borderRadius: 3,
    backgroundColor: 'rgba(255,255,255,0.3)',
    overflow: 'hidden',
  },
  fill: {
    height: 6,
    backgroundColor: colors.white,
  },
});

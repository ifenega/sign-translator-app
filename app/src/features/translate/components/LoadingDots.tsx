import { useEffect, useState } from 'react';
import { Animated, StyleSheet, View } from 'react-native';

import { brightCycle } from '@/shared/theme';

import { cycleColor, useColorCycle } from '../hooks/useColorCycle';

const DOT_COUNT = 3;
const HOP_MS = 1200; // one full hop cycle for all three dots
const STAGGER_MS = 150; // each dot starts a little after the one before

// Three dots that hop one after another while changing colour.
export function LoadingDots() {
  const colorProgress = useColorCycle(brightCycle.length, 3000);
  const [hops] = useState(() => Array.from({ length: DOT_COUNT }, () => new Animated.Value(0)));

  useEffect(() => {
    const loops = hops.map((hop, index) => {
      // Every loop lasts HOP_MS in total, so the dots stay in step.
      const loop = Animated.loop(
        Animated.sequence([
          Animated.delay(index * STAGGER_MS),
          Animated.timing(hop, { toValue: -10, duration: HOP_MS * 0.3, useNativeDriver: false }),
          Animated.timing(hop, { toValue: 0, duration: HOP_MS * 0.3, useNativeDriver: false }),
          Animated.delay(HOP_MS * 0.4 - index * STAGGER_MS),
        ]),
      );
      loop.start();
      return loop;
    });
    return () => loops.forEach((loop) => loop.stop());
  }, [hops]);

  return (
    <View style={styles.row} accessibilityRole="progressbar" accessibilityLabel="Loading">
      {hops.map((hop, index) => (
        <Animated.View
          key={index}
          style={[
            styles.dot,
            {
              backgroundColor: cycleColor(colorProgress, brightCycle, index),
              transform: [{ translateY: hop }],
            },
          ]}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    height: 56,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  dot: {
    width: 18,
    height: 18,
    borderRadius: 9,
  },
});

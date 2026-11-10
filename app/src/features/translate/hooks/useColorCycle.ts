import { useEffect, useState } from 'react';
import { Animated, Easing } from 'react-native';

// Returns a value that runs from 0 to `colorCount` over `durationMs`, forever.
// Interpolate it into a colour to make something slowly change colour.
export function useColorCycle(colorCount: number, durationMs: number) {
  const [progress] = useState(() => new Animated.Value(0));

  useEffect(() => {
    const loop = Animated.loop(
      Animated.timing(progress, {
        toValue: colorCount,
        duration: durationMs,
        easing: Easing.linear,
        useNativeDriver: false, // colours can't use the native driver
      }),
    );
    loop.start();
    return () => loop.stop();
  }, [progress, colorCount, durationMs]);

  return progress;
}

// Turns the cycle value into a colour. `offset` shifts where in the cycle this
// particular item is, so several items can be at different colours.
export function cycleColor(progress: Animated.Value, colorList: string[], offset = 0) {
  const steps = colorList.map((_, index) => index);
  const shifted = colorList.map((_, index) => colorList[(index + offset) % colorList.length]);
  return progress.interpolate({
    inputRange: [...steps, colorList.length],
    outputRange: [...shifted, shifted[0]],
  });
}

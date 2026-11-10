import { useEffect, useState, type ReactNode } from 'react';
import { Animated, Easing } from 'react-native';

type Props = {
  children: ReactNode;
};

// Fades its content in while sliding it up a little, the first time it shows.
export function Appear({ children }: Props) {
  const [progress] = useState(() => new Animated.Value(0));

  useEffect(() => {
    Animated.timing(progress, {
      toValue: 1,
      duration: 380,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  }, [progress]);

  const translateY = progress.interpolate({ inputRange: [0, 1], outputRange: [16, 0] });

  return (
    <Animated.View style={{ opacity: progress, transform: [{ translateY }] }}>{children}</Animated.View>
  );
}

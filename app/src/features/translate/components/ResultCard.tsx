import { useEffect, useState } from 'react';
import { Animated, StyleSheet, Text } from 'react-native';

import { colors, fonts } from '@/shared/theme';

import { Card } from './Card';

type Props = {
  text: string;
};

// One word sits on the glowing accent card as a headline; a sentence reads as
// plain text on a white card, a little smaller when it gets long. Either way
// the card pops in with a small bounce.
export function ResultCard({ text }: Props) {
  const isOneWord = !/\s/.test(text.trim());
  const isLong = text.length > 80;

  const [scale] = useState(() => new Animated.Value(0.9));
  useEffect(() => {
    Animated.spring(scale, { toValue: 1, friction: 5, tension: 120, useNativeDriver: true }).start();
  }, [scale]);

  return (
    <Animated.View style={{ transform: [{ scale }] }}>
      {isOneWord ? (
        <Card style={[styles.card, styles.wordCard]}>
          <Text style={styles.word}>{text}</Text>
        </Card>
      ) : (
        <Card style={styles.card}>
          <Text style={[styles.sentence, isLong && styles.longSentence]}>{text}</Text>
        </Card>
      )}
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  card: {
    paddingVertical: 28,
    paddingHorizontal: 24,
  },
  wordCard: {
    paddingVertical: 36,
    backgroundColor: colors.accent,
    borderColor: 'rgba(18,20,18,0.12)',
    boxShadow: '0 12px 40px rgba(170,220,60,0.55)',
  },
  word: {
    fontFamily: fonts.semibold,
    fontSize: 32,
    lineHeight: 42,
    textAlign: 'center',
    color: colors.text,
  },
  sentence: {
    fontFamily: fonts.regular,
    fontSize: 19,
    lineHeight: 28,
    color: colors.text,
  },
  longSentence: {
    fontSize: 17,
    lineHeight: 25,
  },
});

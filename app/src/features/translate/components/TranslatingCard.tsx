import { StyleSheet, Text } from 'react-native';

import { colors, fonts } from '@/shared/theme';

import { Card } from './Card';
import { LoadingDots } from './LoadingDots';

export function TranslatingCard() {
  return (
    <Card style={styles.card}>
      <LoadingDots />
      <Text style={styles.text} accessibilityLiveRegion="polite">
        Translating…
      </Text>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    paddingVertical: 28,
    paddingHorizontal: 24,
    alignItems: 'center',
    gap: 12,
  },
  text: {
    fontFamily: fonts.semibold,
    fontSize: 17,
    color: colors.text,
  },
});

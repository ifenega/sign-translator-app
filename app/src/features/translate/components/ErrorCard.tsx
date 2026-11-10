import Feather from '@expo/vector-icons/Feather';
import { StyleSheet, Text, View } from 'react-native';

import { colors, fonts } from '@/shared/theme';

import { Card } from './Card';

export function ErrorCard() {
  return (
    <Card style={styles.card}>
      <View style={styles.icon}>
        <Feather name="alert-circle" size={24} color={colors.textMuted} />
      </View>
      <Text style={styles.title}>Couldn’t translate this video</Text>
      <Text style={styles.subtitle}>Keep your hands in view and try again</Text>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    paddingVertical: 28,
    paddingHorizontal: 24,
    alignItems: 'center',
    gap: 8,
  },
  icon: {
    width: 56,
    height: 56,
    marginBottom: 6,
    borderRadius: 28,
    backgroundColor: colors.background,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontFamily: fonts.semibold,
    fontSize: 19,
    textAlign: 'center',
    color: colors.text,
  },
  subtitle: {
    fontFamily: fonts.regular,
    fontSize: 15,
    textAlign: 'center',
    color: colors.textMuted,
  },
});

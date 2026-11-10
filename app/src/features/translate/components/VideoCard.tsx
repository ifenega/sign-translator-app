import Feather from '@expo/vector-icons/Feather';
import { useVideoPlayer, VideoView } from 'expo-video';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { colors, fonts } from '@/shared/theme';

import { Card } from './Card';

type Props = {
  uri: string;
  name: string;
  removeLabel: string;
  onRemove: () => void;
};

// The chosen video, playing on a silent loop, with an ✕ that clears it.
export function VideoCard({ uri, name, removeLabel, onRemove }: Props) {
  const player = useVideoPlayer(uri, (newPlayer) => {
    newPlayer.loop = true;
    newPlayer.muted = true;
    newPlayer.play();
  });

  return (
    <Card style={styles.card}>
      <View style={styles.videoFrame}>
        <VideoView player={player} style={StyleSheet.absoluteFill} contentFit="cover" nativeControls={false} />
        <Pressable
          onPress={onRemove}
          accessibilityRole="button"
          accessibilityLabel={removeLabel}
          hitSlop={8}
          style={({ pressed }) => [styles.removeButton, pressed && styles.removeButtonPressed]}
        >
          <Feather name="x" size={18} color={colors.white} />
        </Pressable>
      </View>
      <Text style={styles.name} numberOfLines={1}>
        {name}
      </Text>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: 10,
    paddingBottom: 14,
    gap: 12,
  },
  videoFrame: {
    height: 220,
    borderRadius: 20,
    overflow: 'hidden',
    backgroundColor: colors.videoTile,
  },
  removeButton: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0,0,0,0.45)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  removeButtonPressed: {
    backgroundColor: 'rgba(0,0,0,0.65)',
  },
  name: {
    paddingHorizontal: 6,
    fontFamily: fonts.medium,
    fontSize: 14,
    color: colors.textMuted,
  },
});

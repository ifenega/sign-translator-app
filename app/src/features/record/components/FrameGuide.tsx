import { StyleSheet, View } from 'react-native';

// A head-and-shoulders outline that shows where to stand, so the hands and
// face stay in the shot.
export function FrameGuide() {
  return (
    <View style={styles.guide}>
      <View style={styles.head} />
      <View style={styles.shoulders} />
    </View>
  );
}

const LINE = 'rgba(255,255,255,0.85)';

const styles = StyleSheet.create({
  guide: {
    alignItems: 'center',
    pointerEvents: 'none',
  },
  head: {
    width: 130,
    height: 160,
    borderRadius: 80,
    borderWidth: 3,
    borderColor: LINE,
  },
  shoulders: {
    width: 300,
    height: 150,
    marginTop: 12,
    borderTopLeftRadius: 150,
    borderTopRightRadius: 150,
    borderWidth: 3,
    borderBottomWidth: 0,
    borderColor: LINE,
  },
});

import { useState } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { RecordModal, type RecordedVideo } from '@/features/record';
import { colors, fonts } from '@/shared/theme';

import { Appear } from '../components/Appear';
import { ErrorCard } from '../components/ErrorCard';
import { ResultCard } from '../components/ResultCard';
import { TranslatingCard } from '../components/TranslatingCard';
import { UploadCard } from '../components/UploadCard';
import { VideoCard } from '../components/VideoCard';
import { useTranslation } from '../hooks/useTranslation';
import { pickVideo } from '../utils/pickVideo';

// The whole app for now: record or upload a video, then see it translated underneath.
export function TranslateScreen() {
  const { state, translate, clear } = useTranslation();
  const [isCameraOpen, setIsCameraOpen] = useState(false);

  async function chooseVideo() {
    const video = await pickVideo();
    if (video) {
      translate(video);
    }
  }

  function translateRecording(video: RecordedVideo) {
    setIsCameraOpen(false);
    translate({ uri: video.uri, name: 'New recording' });
  }

  return (
    <SafeAreaView style={styles.screen} edges={['top', 'bottom']}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>Translate sign language</Text>

        {/* The cards sit in the middle of the space below the title. Each one
            is keyed, so it animates in whenever it first appears. */}
        <View style={styles.body}>
          {state.status === 'idle' ? (
            <Appear key="upload">
              <UploadCard onRecordVideo={() => setIsCameraOpen(true)} onChooseVideo={chooseVideo} />
            </Appear>
          ) : (
            <>
              <Appear key={state.video.uri}>
                <VideoCard
                  uri={state.video.uri}
                  name={state.video.name}
                  removeLabel={state.status === 'translating' ? 'Cancel' : 'Remove video'}
                  onRemove={clear}
                />
              </Appear>
              <Appear key={state.status}>
                {state.status === 'translating' && <TranslatingCard />}
                {state.status === 'done' && <ResultCard text={state.text} />}
                {state.status === 'failed' && <ErrorCard />}
              </Appear>
            </>
          )}
        </View>
      </ScrollView>

      <RecordModal
        visible={isCameraOpen}
        onRecorded={translateRecording}
        onClose={() => setIsCameraOpen(false)}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    flexGrow: 1,
    paddingTop: 24,
    paddingHorizontal: 20,
    paddingBottom: 32,
    gap: 20,
  },
  body: {
    flex: 1,
    justifyContent: 'center',
    gap: 20,
  },
  title: {
    paddingHorizontal: 4,
    fontFamily: fonts.bold,
    fontSize: 28,
    lineHeight: 34,
    letterSpacing: -0.5,
    color: colors.text,
  },
});

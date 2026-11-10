import * as DocumentPicker from 'expo-document-picker';
import * as ImagePicker from 'expo-image-picker';
import { Alert, Platform } from 'react-native';

import type { PickedVideo } from '../types';

// Lets the user choose a video and returns it, or null if they back out.
// Android opens the file browser, so videos anywhere on the phone (Downloads,
// other folders, the gallery) can be chosen. iPhone opens Photos, where its
// videos live.
export async function pickVideo(): Promise<PickedVideo | null> {
  return Platform.OS === 'android' ? pickFromFiles() : pickFromPhotos();
}

async function pickFromFiles(): Promise<PickedVideo | null> {
  const result = await DocumentPicker.getDocumentAsync({ type: 'video/*' });
  if (result.canceled) {
    return null;
  }

  const video = result.assets[0];
  return { uri: video.uri, name: video.name };
}

async function pickFromPhotos(): Promise<PickedVideo | null> {
  const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (!permission.granted) {
    Alert.alert('No access to videos', 'Allow access to your videos in Settings to choose one.');
    return null;
  }

  const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['videos'] });
  if (result.canceled) {
    return null;
  }

  const video = result.assets[0];
  return { uri: video.uri, name: video.fileName ?? 'Video' };
}

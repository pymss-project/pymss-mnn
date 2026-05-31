# iOS Smoke Runner

Minimal signed iPhone app for validating the C++ MNN RoFormer runtime on device.
It bundles one `.mnn` core model, one metadata json file, and one fallback WAV
input. Choose imports an audio file from Files, Run starts `RoformerSeparator`
with Metal/Fused attention, Play previews the first generated WAV, and Share
exports `summary.txt` plus generated WAV files. Selected audio is decoded with
AVFoundation and converted to the model sample rate as stereo float before it
enters the C++ MNN runtime.

Configure the app target from `runtime/mnn` after building/installing an iOS
MNN SDK:

```sh
cmake -S runtime/mnn -B /Volumes/2T/cache/ios_build/pymss-mnn-ios \
  -G Xcode \
  -DCMAKE_TOOLCHAIN_FILE=runtime/MNN-src/cmake/ios.toolchain.cmake \
  -DPLATFORM=OS64 \
  -DDEPLOYMENT_TARGET=16.0 \
  -DMNN_ROOT=/Volumes/2T/cache/ios_build/MNN-ios-install \
  -DMSS_MNN_BUILD_APPS=OFF \
  -DMSS_MNN_BUILD_IOS_SMOKE=ON \
  -DMSS_MNN_IOS_CORE_MODEL=/path/to/core.mnn \
  -DMSS_MNN_IOS_METADATA=/path/to/metadata.json \
  -DMSS_MNN_IOS_INPUT_WAV=/path/to/test_10s.wav \
  -DMSS_MNN_IOS_BUNDLE_ID=com.example.pymssmnn.smoke \
  -DMSS_MNN_IOS_DEVELOPMENT_TEAM=YOURTEAMID
```

Compile without signing first to catch C++ and link errors:

```sh
xcodebuild -project /Volumes/2T/cache/ios_build/pymss-mnn-ios/mss_mnn_runtime.xcodeproj \
  -scheme mss_mnn_ios_smoke \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

Deploy to a connected, unlocked iPhone after Xcode has a valid Apple
development account for `YOURTEAMID`:

```sh
xcodebuild -project /Volumes/2T/cache/ios_build/pymss-mnn-ios/mss_mnn_runtime.xcodeproj \
  -scheme mss_mnn_ios_smoke \
  -configuration Release \
  -destination 'id=DEVICE_UDID' \
  -allowProvisioningUpdates \
  build
```

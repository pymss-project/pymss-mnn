#import "AppDelegate.h"

#import <AVFoundation/AVFoundation.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include "mss_mnn/audio.hpp"
#include "mss_mnn/roformer_separator.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>

#include <mach/mach.h>
#include <mach/task_info.h>
#include <mach/thread_act.h>
#include <mach/thread_info.h>

#ifndef MSS_MNN_IOS_CORE_BUNDLE_NAME
#define MSS_MNN_IOS_CORE_BUNDLE_NAME "core.mnn"
#endif
#ifndef MSS_MNN_IOS_SEGMENT_BUNDLE_NAME
#define MSS_MNN_IOS_SEGMENT_BUNDLE_NAME ""
#endif
#ifndef MSS_MNN_IOS_METADATA_BUNDLE_NAME
#define MSS_MNN_IOS_METADATA_BUNDLE_NAME "metadata.json"
#endif
#ifndef MSS_MNN_IOS_INPUT_BUNDLE_NAME
#define MSS_MNN_IOS_INPUT_BUNDLE_NAME "input.wav"
#endif

@interface AppDelegate () <UIDocumentPickerDelegate, AVAudioPlayerDelegate>
@property(strong, nonatomic) UILabel* statusLabel;
@property(strong, nonatomic) UILabel* modelLabel;
@property(strong, nonatomic) UILabel* inputLabel;
@property(strong, nonatomic) UILabel* metricsLabel;
@property(strong, nonatomic) UILabel* progressLabel;
@property(strong, nonatomic) UILabel* monitorLabel;
@property(strong, nonatomic) UILabel* outputLabel;
@property(strong, nonatomic) UITextView* logView;
@property(strong, nonatomic) UIButton* selectButton;
@property(strong, nonatomic) UIButton* inputPlayButton;
@property(strong, nonatomic) UIButton* runButton;
@property(strong, nonatomic) UIButton* playButton;
@property(strong, nonatomic) UIButton* shareButton;
@property(strong, nonatomic) UIButton* diagnosticsButton;
@property(strong, nonatomic) UIActivityIndicatorView* activityView;
@property(strong, nonatomic) UIProgressView* progressView;
@property(strong, nonatomic) UISlider* inputPlaybackSlider;
@property(strong, nonatomic) UILabel* inputPlaybackTimeLabel;
@property(strong, nonatomic) UISlider* playbackSlider;
@property(strong, nonatomic) UILabel* playbackTimeLabel;
@property(strong, nonatomic) UISegmentedControl* outputSelector;
@property(strong, nonatomic) NSMutableArray<NSURL*>* outputURLs;
@property(strong, nonatomic) NSMutableArray<NSURL*>* audioOutputURLs;
@property(strong, nonatomic) NSMutableArray<NSString*>* audioOutputNames;
@property(strong, nonatomic) NSURL* selectedInputURL;
@property(strong, nonatomic) AVAudioPlayer* inputAudioPlayer;
@property(strong, nonatomic) AVAudioPlayer* audioPlayer;
@property(strong, nonatomic) NSTimer* monitorTimer;
@property(strong, nonatomic) NSTimer* inputPlaybackTimer;
@property(strong, nonatomic) NSTimer* playbackTimer;
@property(assign, nonatomic) BOOL running;
@property(assign, nonatomic) BOOL exitAfterRun;
@property(assign, nonatomic) BOOL diagnosticsVisible;
@property(assign, nonatomic) BOOL inputPlaybackScrubbing;
@property(assign, nonatomic) BOOL playbackScrubbing;
@property(strong, nonatomic) NSString* currentRunID;
@property(strong, nonatomic) NSString* emptyOutputMessage;
@property(assign, nonatomic) uint64_t peakResidentBytes;
@end

namespace {

using Clock = std::chrono::steady_clock;

NSString* BundlePath(const char* name) {
    NSString* resource = [NSString stringWithUTF8String:name];
    NSString* path = [[NSBundle mainBundle] pathForResource:resource ofType:nil];
    if (path == nil) {
        @throw [NSException exceptionWithName:@"MissingResource"
                                       reason:[NSString stringWithFormat:@"missing bundled resource: %@", resource]
                                     userInfo:nil];
    }
    return path;
}

NSString* OptionalBundlePath(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        return nil;
    }
    return BundlePath(name);
}

NSString* DocumentsPath(NSString* name) {
    NSArray<NSURL*>* urls = [[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory
                                                                   inDomains:NSUserDomainMask];
    return [[urls.firstObject URLByAppendingPathComponent:name] path];
}

std::string ToString(NSString* value) {
    return std::string([value UTF8String]);
}

double ElapsedSeconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void WriteTextFile(NSString* path, const std::string& text) {
    std::ofstream out(ToString(path), std::ios::binary);
    out << text;
}

std::string NSErrorText(NSError* error) {
    if (!error) {
        return "unknown error";
    }
    return ToString(error.localizedDescription ?: @"unknown error");
}

NSString* DisplayNameForStem(NSString* stem) {
    NSString* normalized = [[stem stringByReplacingOccurrencesOfString:@"_" withString:@" "] lowercaseString];
    NSDictionary<NSString*, NSString*>* names = @{
        @"vocals": @"人声",
        @"vocal": @"人声",
        @"instrumental": @"伴奏",
        @"accompaniment": @"伴奏",
        @"drums": @"鼓组",
        @"bass": @"贝斯",
        @"other": @"其他",
        @"piano": @"钢琴",
        @"guitar": @"吉他",
    };
    NSString* mapped = names[normalized];
    if (mapped.length > 0) {
        return mapped;
    }
    return stem.length > 0 ? stem : @"音轨";
}

bool IsVocalStemName(const std::string& stem) {
    std::string normalized = stem;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return normalized == "vocals" || normalized == "vocal";
}

mss_mnn::AudioBuffer MakeResidualAudio(const mss_mnn::AudioBuffer& mix, const mss_mnn::AudioBuffer& stem) {
    mss_mnn::AudioBuffer residual = mix;
    const std::size_t frames = std::min(mix.frames(), stem.frames());
    const int channels = std::min(mix.channels, stem.channels);
    for (int ch = 0; ch < channels; ++ch) {
        const std::size_t mixBase = static_cast<std::size_t>(ch) * mix.frames();
        const std::size_t stemBase = static_cast<std::size_t>(ch) * stem.frames();
        for (std::size_t frame = 0; frame < frames; ++frame) {
            residual.data[mixBase + frame] = mix.data[mixBase + frame] - stem.data[stemBase + frame];
        }
    }
    return residual;
}

NSString* FormatPlaybackTime(NSTimeInterval seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return @"--:--";
    }
    const NSInteger totalSeconds = static_cast<NSInteger>(std::round(seconds));
    const NSInteger minutes = totalSeconds / 60;
    const NSInteger remainingSeconds = totalSeconds % 60;
    return [NSString stringWithFormat:@"%02ld:%02ld", static_cast<long>(minutes), static_cast<long>(remainingSeconds)];
}

void WriteProgress(const std::string& text) {
    WriteTextFile(DocumentsPath(@"summary.txt"), text + "\n");
}

NSString* LastPickerDirectoryBookmarkKey() {
    return @"mss_mnn.last_picker_directory_bookmark";
}

bool IsInsideAppContainer(NSURL* url) {
    NSString* path = url.path;
    NSString* home = NSHomeDirectory();
    return path.length > 0 && home.length > 0 &&
           ([path isEqualToString:home] || [path hasPrefix:[home stringByAppendingString:@"/"]]);
}

NSURL* LastPickerDirectoryURL() {
    NSData* data = [[NSUserDefaults standardUserDefaults] dataForKey:LastPickerDirectoryBookmarkKey()];
    if (!data) {
        return nil;
    }
    BOOL stale = NO;
    NSError* error = nil;
    NSURL* url = [NSURL URLByResolvingBookmarkData:data
                                           options:0
                                     relativeToURL:nil
                               bookmarkDataIsStale:&stale
                                             error:&error];
    if (!url || stale || error || IsInsideAppContainer(url)) {
        [[NSUserDefaults standardUserDefaults] removeObjectForKey:LastPickerDirectoryBookmarkKey()];
        return nil;
    }
    return url;
}

void SaveLastPickerDirectoryURL(NSURL* url) {
    if (!url) {
        return;
    }
    if (IsInsideAppContainer(url)) {
        [[NSUserDefaults standardUserDefaults] removeObjectForKey:LastPickerDirectoryBookmarkKey()];
        return;
    }
    NSError* error = nil;
    NSData* data = [url bookmarkDataWithOptions:0
                includingResourceValuesForKeys:nil
                                 relativeToURL:nil
                                         error:&error];
    if (data && !error) {
        [[NSUserDefaults standardUserDefaults] setObject:data forKey:LastPickerDirectoryBookmarkKey()];
    }
}

double CurrentProcessCPUPercent() {
    thread_array_t threads = nullptr;
    mach_msg_type_number_t threadCount = 0;
    const kern_return_t threadStatus = task_threads(mach_task_self(), &threads, &threadCount);
    if (threadStatus != KERN_SUCCESS) {
        return -1.0;
    }

    double cpu = 0.0;
    for (mach_msg_type_number_t i = 0; i < threadCount; ++i) {
        thread_info_data_t info;
        mach_msg_type_number_t infoCount = THREAD_INFO_MAX;
        const kern_return_t infoStatus = thread_info(threads[i], THREAD_BASIC_INFO, info, &infoCount);
        if (infoStatus == KERN_SUCCESS) {
            const auto* basic = reinterpret_cast<thread_basic_info_t>(info);
            if ((basic->flags & TH_FLAGS_IDLE) == 0) {
                cpu += static_cast<double>(basic->cpu_usage) / TH_USAGE_SCALE * 100.0;
            }
        }
        mach_port_deallocate(mach_task_self(), threads[i]);
    }

    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads), threadCount * sizeof(thread_t));
    return cpu;
}

uint64_t CurrentResidentBytes() {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t status =
        task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count);
    if (status != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<uint64_t>(info.resident_size);
}

NSString* FormatBytes(uint64_t bytes) {
    const double value = static_cast<double>(bytes);
    const double gib = 1024.0 * 1024.0 * 1024.0;
    const double mib = 1024.0 * 1024.0;
    if (value >= gib) {
        return [NSString stringWithFormat:@"%.2f GiB", value / gib];
    }
    return [NSString stringWithFormat:@"%.1f MiB", value / mib];
}

mss_mnn::AudioBuffer ReadAudioURL(NSURL* url, int sampleRate, int channels) {
    NSError* error = nil;
    AVAudioFile* file = [[AVAudioFile alloc] initForReading:url error:&error];
    if (!file) {
        throw std::runtime_error("failed to open audio: " + NSErrorText(error));
    }

    AVAudioFormat* inputFormat = file.processingFormat;
    if (file.length <= 0 || file.length > static_cast<AVAudioFramePosition>(std::numeric_limits<AVAudioFrameCount>::max())) {
        throw std::runtime_error("unsupported audio length");
    }
    AVAudioPCMBuffer* inputBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:inputFormat
                                                                  frameCapacity:static_cast<AVAudioFrameCount>(file.length)];
    if (![file readIntoBuffer:inputBuffer error:&error]) {
        throw std::runtime_error("failed to read audio: " + NSErrorText(error));
    }

    AVAudioFormat* outputFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                                                   sampleRate:sampleRate
                                                                     channels:static_cast<AVAudioChannelCount>(channels)
                                                                  interleaved:NO];
    AVAudioConverter* converter = [[AVAudioConverter alloc] initFromFormat:inputFormat toFormat:outputFormat];
    if (!converter) {
        throw std::runtime_error("failed to create audio converter");
    }

    const double ratio = static_cast<double>(sampleRate) / inputFormat.sampleRate;
    const auto capacity = static_cast<AVAudioFrameCount>(std::ceil(inputBuffer.frameLength * ratio) + 4096.0);
    AVAudioPCMBuffer* outputBuffer = [[AVAudioPCMBuffer alloc] initWithPCMFormat:outputFormat frameCapacity:capacity];
    __block BOOL providedInput = NO;
    AVAudioConverterInputBlock inputBlock = ^AVAudioBuffer* _Nullable(AVAudioPacketCount, AVAudioConverterInputStatus* outStatus) {
        if (providedInput) {
            *outStatus = AVAudioConverterInputStatus_EndOfStream;
            return nil;
        }
        providedInput = YES;
        *outStatus = AVAudioConverterInputStatus_HaveData;
        return inputBuffer;
    };
    const AVAudioConverterOutputStatus status = [converter convertToBuffer:outputBuffer error:&error withInputFromBlock:inputBlock];
    if (status == AVAudioConverterOutputStatus_Error) {
        throw std::runtime_error("failed to convert audio: " + NSErrorText(error));
    }
    if (!outputBuffer.floatChannelData || outputBuffer.frameLength == 0) {
        throw std::runtime_error("audio converter returned empty output");
    }

    mss_mnn::AudioBuffer audio;
    audio.sample_rate = sampleRate;
    audio.channels = channels;
    const auto frames = static_cast<std::size_t>(outputBuffer.frameLength);
    audio.data.assign(static_cast<std::size_t>(channels) * frames, 0.0f);
    for (int ch = 0; ch < channels; ++ch) {
        const float* src = outputBuffer.floatChannelData[ch];
        std::copy(src, src + frames, audio.data.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(ch) * frames));
    }
    return audio;
}

}  // namespace

@implementation AppDelegate

- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)launchOptions {
    self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];

    UIViewController* controller = [[UIViewController alloc] init];
    controller.view.backgroundColor = UIColor.systemBackgroundColor;
    self.outputURLs = [NSMutableArray array];
    self.audioOutputURLs = [NSMutableArray array];
    self.audioOutputNames = [NSMutableArray array];
    self.emptyOutputMessage = @"还没有输出音轨";
    [self configureUIInView:controller.view];
    [[AVAudioSession sharedInstance] setCategory:AVAudioSessionCategoryPlayback error:nil];
    [[AVAudioSession sharedInstance] setActive:YES error:nil];

    self.window.rootViewController = controller;
    [self.window makeKeyAndVisible];

    NSDictionary<NSString*, NSString*>* environment = NSProcessInfo.processInfo.environment;
    NSArray<NSString*>* arguments = NSProcessInfo.processInfo.arguments;
    self.exitAfterRun = [environment[@"MSS_MNN_IOS_EXIT_AFTER_RUN"] isEqualToString:@"1"] ||
                        [arguments containsObject:@"--exit-after-run"];
    if ([environment[@"MSS_MNN_IOS_AUTORUN"] isEqualToString:@"1"] ||
        [arguments containsObject:@"--autorun"]) {
        [self runSmokeTest];
    }
    return YES;
}

- (UILabel*)makeLabelWithFont:(UIFont*)font color:(UIColor*)color lines:(NSInteger)lines {
    UILabel* label = [[UILabel alloc] init];
    label.translatesAutoresizingMaskIntoConstraints = NO;
    label.font = font;
    label.textColor = color;
    label.numberOfLines = lines;
    return label;
}

- (UIButton*)makeButtonWithTitle:(NSString*)title
                  systemImageName:(NSString*)systemImageName
                           filled:(BOOL)filled
                           action:(SEL)action {
    UIButton* button = [UIButton buttonWithType:UIButtonTypeSystem];
    button.translatesAutoresizingMaskIntoConstraints = NO;
    UIButtonConfiguration* configuration = filled ? [UIButtonConfiguration filledButtonConfiguration]
                                                  : [UIButtonConfiguration tintedButtonConfiguration];
    configuration.title = title;
    configuration.cornerStyle = UIButtonConfigurationCornerStyleMedium;
    configuration.contentInsets = NSDirectionalEdgeInsetsMake(12.0, 16.0, 12.0, 16.0);
    if (systemImageName.length > 0) {
        configuration.image = [UIImage systemImageNamed:systemImageName];
        configuration.imagePadding = 8.0;
    }
    if (filled) {
        configuration.baseBackgroundColor = UIColor.labelColor;
        configuration.baseForegroundColor = UIColor.systemBackgroundColor;
    }
    button.configuration = configuration;
    button.titleLabel.font = [UIFont systemFontOfSize:16.0 weight:UIFontWeightSemibold];
    [button addTarget:self action:action forControlEvents:UIControlEventTouchUpInside];
    [button.heightAnchor constraintGreaterThanOrEqualToConstant:48.0].active = YES;
    return button;
}

- (UIView*)makePanel {
    UIView* panel = [[UIView alloc] init];
    panel.translatesAutoresizingMaskIntoConstraints = NO;
    panel.backgroundColor = UIColor.secondarySystemGroupedBackgroundColor;
    panel.layer.cornerRadius = 8.0;
    panel.layer.masksToBounds = YES;
    panel.layer.borderWidth = 1.0 / UIScreen.mainScreen.scale;
    panel.layer.borderColor = UIColor.separatorColor.CGColor;
    panel.layoutMargins = UIEdgeInsetsMake(16.0, 16.0, 16.0, 16.0);
    return panel;
}

- (UIStackView*)makePanelStackInPanel:(UIView*)panel {
    UIStackView* stack = [[UIStackView alloc] init];
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    stack.axis = UILayoutConstraintAxisVertical;
    stack.spacing = 12.0;
    [panel addSubview:stack];
    UILayoutGuide* margins = panel.layoutMarginsGuide;
    [NSLayoutConstraint activateConstraints:@[
        [stack.leadingAnchor constraintEqualToAnchor:margins.leadingAnchor],
        [stack.trailingAnchor constraintEqualToAnchor:margins.trailingAnchor],
        [stack.topAnchor constraintEqualToAnchor:margins.topAnchor],
        [stack.bottomAnchor constraintEqualToAnchor:margins.bottomAnchor],
    ]];
    return stack;
}

- (UILabel*)makeSectionLabel:(NSString*)text {
    UILabel* label = [self makeLabelWithFont:[UIFont systemFontOfSize:12.0 weight:UIFontWeightSemibold]
                                       color:UIColor.secondaryLabelColor
                                       lines:1];
    label.text = text;
    return label;
}

- (void)styleStatusForText:(NSString*)status {
    NSString* displayStatus = status;
    if ([status isEqualToString:@"Idle"]) {
        displayStatus = @"空闲";
    } else if ([status isEqualToString:@"Ready"]) {
        displayStatus = @"就绪";
    } else if ([status isEqualToString:@"Running"]) {
        displayStatus = @"分离中";
    } else if ([status isEqualToString:@"Complete"]) {
        displayStatus = @"完成";
    } else if ([status isEqualToString:@"Failed"] || [status containsString:@"failed"]) {
        displayStatus = @"失败";
    }
    self.statusLabel.text = displayStatus;
    UIColor* foreground = UIColor.secondaryLabelColor;
    UIColor* background = UIColor.tertiarySystemFillColor;
    if ([status isEqualToString:@"Running"]) {
        foreground = UIColor.systemBlueColor;
        background = UIColor.systemBlueColor;
        background = [background colorWithAlphaComponent:0.14];
    } else if ([status isEqualToString:@"Complete"] || [status isEqualToString:@"Ready"]) {
        foreground = UIColor.systemGreenColor;
        background = UIColor.systemGreenColor;
        background = [background colorWithAlphaComponent:0.14];
    } else if ([status isEqualToString:@"Failed"] || [status containsString:@"failed"]) {
        foreground = UIColor.systemRedColor;
        background = UIColor.systemRedColor;
        background = [background colorWithAlphaComponent:0.14];
    }
    self.statusLabel.textColor = foreground;
    self.statusLabel.backgroundColor = background;
}

- (void)configureUIInView:(UIView*)view {
    UILayoutGuide* guide = view.safeAreaLayoutGuide;
    view.backgroundColor = UIColor.systemGroupedBackgroundColor;

    UIScrollView* scrollView = [[UIScrollView alloc] init];
    scrollView.translatesAutoresizingMaskIntoConstraints = NO;
    scrollView.alwaysBounceVertical = YES;
    [view addSubview:scrollView];

    UIStackView* stack = [[UIStackView alloc] init];
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    stack.axis = UILayoutConstraintAxisVertical;
    stack.spacing = 16.0;
    stack.layoutMargins = UIEdgeInsetsMake(20.0, 16.0, 24.0, 16.0);
    stack.layoutMarginsRelativeArrangement = YES;
    [scrollView addSubview:stack];

    UILabel* titleLabel = [self makeLabelWithFont:[UIFont systemFontOfSize:28.0 weight:UIFontWeightBold]
                                            color:UIColor.labelColor
                                            lines:1];
    titleLabel.text = @"MSS MNN";
    UILabel* subtitleLabel = [self makeLabelWithFont:[UIFont systemFontOfSize:14.0 weight:UIFontWeightRegular]
                                               color:UIColor.secondaryLabelColor
                                               lines:1];
    subtitleLabel.text = @"本机音频分离";
    UIStackView* titleStack = [[UIStackView alloc] initWithArrangedSubviews:@[titleLabel, subtitleLabel]];
    titleStack.axis = UILayoutConstraintAxisVertical;
    titleStack.spacing = 2.0;

    self.statusLabel = [self makeLabelWithFont:[UIFont systemFontOfSize:13.0 weight:UIFontWeightSemibold]
                                         color:UIColor.secondaryLabelColor
                                         lines:1];
    self.statusLabel.text = @"空闲";
    self.statusLabel.textAlignment = NSTextAlignmentCenter;
    self.statusLabel.backgroundColor = UIColor.tertiarySystemFillColor;
    self.statusLabel.layer.cornerRadius = 8.0;
    self.statusLabel.layer.masksToBounds = YES;
    [self.statusLabel.widthAnchor constraintGreaterThanOrEqualToConstant:88.0].active = YES;
    [self.statusLabel.heightAnchor constraintEqualToConstant:32.0].active = YES;

    UIStackView* header = [[UIStackView alloc] initWithArrangedSubviews:@[titleStack, self.statusLabel]];
    header.axis = UILayoutConstraintAxisHorizontal;
    header.alignment = UIStackViewAlignmentCenter;
    header.spacing = 12.0;
    [titleStack setContentHuggingPriority:UILayoutPriorityDefaultLow forAxis:UILayoutConstraintAxisHorizontal];
    [self.statusLabel setContentHuggingPriority:UILayoutPriorityRequired forAxis:UILayoutConstraintAxisHorizontal];
    [stack addArrangedSubview:header];

    UIView* inputPanel = [self makePanel];
    UIStackView* inputStack = [self makePanelStackInPanel:inputPanel];
    [inputStack addArrangedSubview:[self makeSectionLabel:@"输入音频"]];
    self.inputLabel = [self makeLabelWithFont:[UIFont systemFontOfSize:19.0 weight:UIFontWeightSemibold]
                                        color:UIColor.labelColor
                                        lines:2];
    self.inputLabel.text = [NSString stringWithFormat:@"内置示例：%@", @MSS_MNN_IOS_INPUT_BUNDLE_NAME];
    self.modelLabel = [self makeLabelWithFont:[UIFont systemFontOfSize:13.0 weight:UIFontWeightRegular]
                                        color:UIColor.secondaryLabelColor
                                        lines:2];
    self.modelLabel.text = [NSString stringWithFormat:@"模型：%@", @MSS_MNN_IOS_CORE_BUNDLE_NAME];
    [inputStack addArrangedSubview:self.inputLabel];
    [inputStack addArrangedSubview:self.modelLabel];
    self.selectButton = [self makeButtonWithTitle:@"选择音频"
                                  systemImageName:@"waveform"
                                           filled:NO
                                           action:@selector(selectButtonTapped)];
    self.inputPlayButton = [self makeButtonWithTitle:@"试听输入"
                                      systemImageName:@"play.fill"
                                               filled:NO
                                               action:@selector(inputPlayButtonTapped)];
    self.inputPlayButton.enabled = YES;
    self.inputPlaybackSlider = [[UISlider alloc] init];
    self.inputPlaybackSlider.translatesAutoresizingMaskIntoConstraints = NO;
    self.inputPlaybackSlider.minimumValue = 0.0f;
    self.inputPlaybackSlider.maximumValue = 1.0f;
    self.inputPlaybackSlider.value = 0.0f;
    self.inputPlaybackSlider.enabled = NO;
    [self.inputPlaybackSlider addTarget:self action:@selector(inputPlaybackSliderTouchDown)
                       forControlEvents:UIControlEventTouchDown];
    [self.inputPlaybackSlider addTarget:self action:@selector(inputPlaybackSliderValueChanged)
                       forControlEvents:UIControlEventValueChanged];
    [self.inputPlaybackSlider addTarget:self action:@selector(inputPlaybackSliderTouchUp)
                       forControlEvents:UIControlEventTouchUpInside | UIControlEventTouchUpOutside | UIControlEventTouchCancel];
    self.inputPlaybackTimeLabel = [self makeLabelWithFont:[UIFont monospacedSystemFontOfSize:13.0 weight:UIFontWeightRegular]
                                                    color:UIColor.secondaryLabelColor
                                                    lines:1];
    self.inputPlaybackTimeLabel.text = @"00:00 / 00:00";
    UIStackView* inputPlaybackStack =
        [[UIStackView alloc] initWithArrangedSubviews:@[self.inputPlaybackSlider, self.inputPlaybackTimeLabel]];
    inputPlaybackStack.axis = UILayoutConstraintAxisVertical;
    inputPlaybackStack.spacing = 6.0;
    [inputStack addArrangedSubview:inputPlaybackStack];
    self.runButton = [self makeButtonWithTitle:@"开始分离"
                               systemImageName:@"play.fill"
                                        filled:YES
                                        action:@selector(runButtonTapped)];
    UIStackView* inputControls = [[UIStackView alloc] initWithArrangedSubviews:@[self.selectButton, self.inputPlayButton]];
    inputControls.axis = UILayoutConstraintAxisHorizontal;
    inputControls.alignment = UIStackViewAlignmentFill;
    inputControls.distribution = UIStackViewDistributionFillEqually;
    inputControls.spacing = 10.0;
    [inputStack addArrangedSubview:inputControls];
    [inputStack addArrangedSubview:self.runButton];
    [stack addArrangedSubview:inputPanel];

    UIView* sessionPanel = [self makePanel];
    UIStackView* sessionStack = [self makePanelStackInPanel:sessionPanel];
    UIStackView* sessionHeader = [[UIStackView alloc] init];
    sessionHeader.axis = UILayoutConstraintAxisHorizontal;
    sessionHeader.alignment = UIStackViewAlignmentCenter;
    sessionHeader.spacing = 10.0;
    UILabel* sessionLabel = [self makeSectionLabel:@"运行状态"];
    self.activityView = [[UIActivityIndicatorView alloc] initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleMedium];
    self.activityView.translatesAutoresizingMaskIntoConstraints = NO;
    self.activityView.hidesWhenStopped = YES;
    [sessionHeader addArrangedSubview:sessionLabel];
    [sessionHeader addArrangedSubview:self.activityView];
    [sessionLabel setContentHuggingPriority:UILayoutPriorityDefaultLow forAxis:UILayoutConstraintAxisHorizontal];
    [self.activityView setContentHuggingPriority:UILayoutPriorityRequired forAxis:UILayoutConstraintAxisHorizontal];
    [sessionStack addArrangedSubview:sessionHeader];
    self.metricsLabel = [self makeLabelWithFont:[UIFont monospacedSystemFontOfSize:15.0 weight:UIFontWeightRegular]
                                          color:UIColor.labelColor
                                          lines:0];
    self.metricsLabel.text = @"就绪\n初始化 --   推理 --\n实时率 --    速度 --";
    [sessionStack addArrangedSubview:self.metricsLabel];
    self.progressLabel = [self makeLabelWithFont:[UIFont systemFontOfSize:13.0 weight:UIFontWeightSemibold]
                                           color:UIColor.secondaryLabelColor
                                           lines:1];
    self.progressLabel.text = @"等待开始  0%";
    self.progressView = [[UIProgressView alloc] initWithProgressViewStyle:UIProgressViewStyleDefault];
    self.progressView.translatesAutoresizingMaskIntoConstraints = NO;
    self.progressView.progress = 0.0f;
    self.progressView.trackTintColor = UIColor.tertiarySystemFillColor;
    self.progressView.progressTintColor = UIColor.systemBlueColor;
    [sessionStack addArrangedSubview:self.progressLabel];
    [sessionStack addArrangedSubview:self.progressView];
    [self.progressView.heightAnchor constraintGreaterThanOrEqualToConstant:4.0].active = YES;
    [sessionStack addArrangedSubview:[self makeSectionLabel:@"资源占用"]];
    self.monitorLabel = [self makeLabelWithFont:[UIFont monospacedSystemFontOfSize:13.0 weight:UIFontWeightRegular]
                                          color:UIColor.secondaryLabelColor
                                          lines:0];
    [sessionStack addArrangedSubview:self.monitorLabel];
    [stack addArrangedSubview:sessionPanel];

    UIView* outputPanel = [self makePanel];
    UIStackView* outputStack = [self makePanelStackInPanel:outputPanel];
    [outputStack addArrangedSubview:[self makeSectionLabel:@"输出音轨"]];
    self.outputLabel = [self makeLabelWithFont:[UIFont systemFontOfSize:18.0 weight:UIFontWeightSemibold]
                                         color:UIColor.labelColor
                                         lines:2];
    self.outputLabel.text = @"还没有输出音轨";
    [outputStack addArrangedSubview:self.outputLabel];
    self.outputSelector = [[UISegmentedControl alloc] initWithItems:@[]];
    self.outputSelector.translatesAutoresizingMaskIntoConstraints = NO;
    self.outputSelector.hidden = YES;
    [self.outputSelector addTarget:self action:@selector(outputSelectionChanged) forControlEvents:UIControlEventValueChanged];
    [outputStack addArrangedSubview:self.outputSelector];
    self.playbackSlider = [[UISlider alloc] init];
    self.playbackSlider.translatesAutoresizingMaskIntoConstraints = NO;
    self.playbackSlider.minimumValue = 0.0f;
    self.playbackSlider.maximumValue = 1.0f;
    self.playbackSlider.value = 0.0f;
    self.playbackSlider.enabled = NO;
    [self.playbackSlider addTarget:self action:@selector(playbackSliderTouchDown)
                  forControlEvents:UIControlEventTouchDown];
    [self.playbackSlider addTarget:self action:@selector(playbackSliderValueChanged)
                  forControlEvents:UIControlEventValueChanged];
    [self.playbackSlider addTarget:self action:@selector(playbackSliderTouchUp)
                  forControlEvents:UIControlEventTouchUpInside | UIControlEventTouchUpOutside | UIControlEventTouchCancel];
    self.playbackTimeLabel = [self makeLabelWithFont:[UIFont monospacedSystemFontOfSize:13.0 weight:UIFontWeightRegular]
                                               color:UIColor.secondaryLabelColor
                                               lines:1];
    self.playbackTimeLabel.text = @"00:00 / 00:00";
    UIStackView* playbackStack = [[UIStackView alloc] initWithArrangedSubviews:@[self.playbackSlider, self.playbackTimeLabel]];
    playbackStack.axis = UILayoutConstraintAxisVertical;
    playbackStack.spacing = 6.0;
    [outputStack addArrangedSubview:playbackStack];
    self.playButton = [self makeButtonWithTitle:@"播放"
                                systemImageName:@"play.fill"
                                         filled:NO
                                         action:@selector(playButtonTapped)];
    self.playButton.enabled = NO;
    self.shareButton = [self makeButtonWithTitle:@"导出"
                                 systemImageName:@"square.and.arrow.up"
                                          filled:NO
                                          action:@selector(shareButtonTapped)];
    self.shareButton.enabled = NO;
    UIStackView* outputControls = [[UIStackView alloc] initWithArrangedSubviews:@[self.playButton, self.shareButton]];
    outputControls.axis = UILayoutConstraintAxisHorizontal;
    outputControls.alignment = UIStackViewAlignmentFill;
    outputControls.distribution = UIStackViewDistributionFillEqually;
    outputControls.spacing = 10.0;
    [outputStack addArrangedSubview:outputControls];
    [stack addArrangedSubview:outputPanel];

    self.diagnosticsButton = [self makeButtonWithTitle:@"诊断信息"
                                       systemImageName:@"stethoscope"
                                                filled:NO
                                                action:@selector(diagnosticsButtonTapped)];
    [stack addArrangedSubview:self.diagnosticsButton];

    self.logView = [[UITextView alloc] init];
    self.logView.translatesAutoresizingMaskIntoConstraints = NO;
    self.logView.editable = NO;
    self.logView.alwaysBounceVertical = YES;
    self.logView.font = [UIFont monospacedSystemFontOfSize:12.0 weight:UIFontWeightRegular];
    self.logView.backgroundColor = UIColor.secondarySystemBackgroundColor;
    self.logView.layer.cornerRadius = 8.0;
    self.logView.layer.masksToBounds = YES;
    self.logView.textContainerInset = UIEdgeInsetsMake(10.0, 10.0, 10.0, 10.0);
    self.logView.text = @"";
    self.logView.hidden = YES;
    [stack addArrangedSubview:self.logView];

    [NSLayoutConstraint activateConstraints:@[
        [scrollView.leadingAnchor constraintEqualToAnchor:guide.leadingAnchor],
        [scrollView.trailingAnchor constraintEqualToAnchor:guide.trailingAnchor],
        [scrollView.topAnchor constraintEqualToAnchor:guide.topAnchor],
        [scrollView.bottomAnchor constraintEqualToAnchor:guide.bottomAnchor],
        [stack.leadingAnchor constraintEqualToAnchor:scrollView.contentLayoutGuide.leadingAnchor],
        [stack.trailingAnchor constraintEqualToAnchor:scrollView.contentLayoutGuide.trailingAnchor],
        [stack.topAnchor constraintEqualToAnchor:scrollView.contentLayoutGuide.topAnchor],
        [stack.bottomAnchor constraintEqualToAnchor:scrollView.contentLayoutGuide.bottomAnchor],
        [stack.widthAnchor constraintEqualToAnchor:scrollView.frameLayoutGuide.widthAnchor],
        [self.logView.heightAnchor constraintEqualToConstant:180.0],
    ]];
    [self refreshOutputControls];
    [self styleStatusForText:@"Idle"];
    [self updateMonitor];
    NSError* inputError = nil;
    if (![self prepareInputAudioPlayerWithStartTime:0.0 error:&inputError]) {
        [self appendLog:[NSString stringWithFormat:@"load input player failed: %@", inputError.localizedDescription ?: @"unknown error"]];
    }
}

- (void)setButton:(UIButton*)button title:(NSString*)title {
    UIButtonConfiguration* configuration = button.configuration ?: [UIButtonConfiguration borderedButtonConfiguration];
    configuration.title = title;
    button.configuration = configuration;
}

- (NSInteger)selectedOutputIndex {
    if (self.audioOutputURLs.count == 0) {
        return NSNotFound;
    }
    NSInteger selected = self.outputSelector.selectedSegmentIndex;
    if (selected < 0 || selected >= static_cast<NSInteger>(self.audioOutputURLs.count)) {
        return 0;
    }
    return selected;
}

- (NSURL*)selectedAudioOutputURL {
    NSInteger index = [self selectedOutputIndex];
    if (index == NSNotFound) {
        return nil;
    }
    return self.audioOutputURLs[static_cast<NSUInteger>(index)];
}

- (NSString*)selectedAudioOutputName {
    NSInteger index = [self selectedOutputIndex];
    if (index == NSNotFound) {
        return nil;
    }
    if (index < static_cast<NSInteger>(self.audioOutputNames.count)) {
        return self.audioOutputNames[static_cast<NSUInteger>(index)];
    }
    return self.audioOutputURLs[static_cast<NSUInteger>(index)].lastPathComponent;
}

- (NSURL*)currentInputAudioURL {
    if (self.selectedInputURL) {
        return self.selectedInputURL;
    }
    return [NSURL fileURLWithPath:BundlePath(MSS_MNN_IOS_INPUT_BUNDLE_NAME)];
}

- (void)stopInputPlaybackTimer {
    [self.inputPlaybackTimer invalidate];
    self.inputPlaybackTimer = nil;
}

- (void)startInputPlaybackTimer {
    if (self.inputPlaybackTimer) {
        return;
    }
    self.inputPlaybackTimer = [NSTimer scheduledTimerWithTimeInterval:0.25
                                                               target:self
                                                             selector:@selector(updateInputPlaybackProgress)
                                                             userInfo:nil
                                                              repeats:YES];
    self.inputPlaybackTimer.tolerance = 0.05;
}

- (void)setInputPlaybackPosition:(NSTimeInterval)current duration:(NSTimeInterval)duration {
    const BOOL hasDuration = std::isfinite(duration) && duration > 0.0;
    self.inputPlaybackSlider.enabled = !self.running && hasDuration;
    self.inputPlaybackSlider.minimumValue = 0.0f;
    self.inputPlaybackSlider.maximumValue = hasDuration ? static_cast<float>(duration) : 1.0f;
    if (!self.inputPlaybackScrubbing) {
        self.inputPlaybackSlider.value = hasDuration ? static_cast<float>(std::max(0.0, std::min(current, duration))) : 0.0f;
    }
    NSString* currentText = FormatPlaybackTime(hasDuration ? current : 0.0);
    NSString* durationText = FormatPlaybackTime(hasDuration ? duration : 0.0);
    self.inputPlaybackTimeLabel.text = [NSString stringWithFormat:@"%@ / %@", currentText, durationText];
}

- (void)resetInputPlaybackProgress {
    [self setInputPlaybackPosition:0.0 duration:0.0];
}

- (void)updateInputPlaybackProgress {
    if (!self.inputAudioPlayer) {
        [self resetInputPlaybackProgress];
        return;
    }
    [self setInputPlaybackPosition:self.inputAudioPlayer.currentTime duration:self.inputAudioPlayer.duration];
}

- (BOOL)prepareInputAudioPlayerWithStartTime:(NSTimeInterval)startTime error:(NSError**)error {
    NSURL* url = [self currentInputAudioURL];
    if (!self.inputAudioPlayer || ![self.inputAudioPlayer.url isEqual:url]) {
        [self stopInputPlaybackTimer];
        self.inputAudioPlayer = [[AVAudioPlayer alloc] initWithContentsOfURL:url error:error];
        if (!self.inputAudioPlayer) {
            [self resetInputPlaybackProgress];
            return NO;
        }
        self.inputAudioPlayer.delegate = self;
        [self.inputAudioPlayer prepareToPlay];
    }
    if (startTime > 0.0 && startTime < self.inputAudioPlayer.duration) {
        self.inputAudioPlayer.currentTime = startTime;
    } else if (startTime <= 0.0) {
        self.inputAudioPlayer.currentTime = 0.0;
    }
    [self updateInputPlaybackProgress];
    return YES;
}

- (void)stopInputPlayback {
    [self stopInputPlaybackTimer];
    [self.inputAudioPlayer stop];
    self.inputAudioPlayer = nil;
    self.inputPlaybackScrubbing = NO;
    [self setButton:self.inputPlayButton title:@"试听输入"];
    [self resetInputPlaybackProgress];
}

- (void)inputPlaybackSliderTouchDown {
    if (!self.inputAudioPlayer) {
        return;
    }
    self.inputPlaybackScrubbing = YES;
}

- (void)inputPlaybackSliderValueChanged {
    if (!self.inputAudioPlayer) {
        return;
    }
    [self setInputPlaybackPosition:self.inputPlaybackSlider.value duration:self.inputAudioPlayer.duration];
}

- (void)inputPlaybackSliderTouchUp {
    if (!self.inputAudioPlayer) {
        self.inputPlaybackScrubbing = NO;
        return;
    }
    self.inputAudioPlayer.currentTime =
        std::max(0.0, std::min(static_cast<double>(self.inputPlaybackSlider.value), self.inputAudioPlayer.duration));
    self.inputPlaybackScrubbing = NO;
    [self updateInputPlaybackProgress];
}

- (void)stopPlaybackTimer {
    [self.playbackTimer invalidate];
    self.playbackTimer = nil;
}

- (void)startPlaybackTimer {
    if (self.playbackTimer) {
        return;
    }
    self.playbackTimer = [NSTimer scheduledTimerWithTimeInterval:0.25
                                                          target:self
                                                        selector:@selector(updatePlaybackProgress)
                                                        userInfo:nil
                                                         repeats:YES];
    self.playbackTimer.tolerance = 0.05;
}

- (void)setPlaybackPosition:(NSTimeInterval)current duration:(NSTimeInterval)duration {
    const BOOL hasDuration = std::isfinite(duration) && duration > 0.0;
    self.playbackSlider.enabled = !self.running && self.audioOutputURLs.count > 0 && hasDuration;
    self.playbackSlider.minimumValue = 0.0f;
    self.playbackSlider.maximumValue = hasDuration ? static_cast<float>(duration) : 1.0f;
    if (!self.playbackScrubbing) {
        self.playbackSlider.value = hasDuration ? static_cast<float>(std::max(0.0, std::min(current, duration))) : 0.0f;
    }
    NSString* currentText = FormatPlaybackTime(hasDuration ? current : 0.0);
    NSString* durationText = FormatPlaybackTime(hasDuration ? duration : 0.0);
    self.playbackTimeLabel.text = [NSString stringWithFormat:@"%@ / %@", currentText, durationText];
}

- (void)resetPlaybackProgress {
    [self setPlaybackPosition:0.0 duration:0.0];
}

- (void)updatePlaybackProgress {
    if (!self.audioPlayer) {
        [self resetPlaybackProgress];
        return;
    }
    [self setPlaybackPosition:self.audioPlayer.currentTime duration:self.audioPlayer.duration];
}

- (BOOL)prepareSelectedAudioPlayerWithStartTime:(NSTimeInterval)startTime error:(NSError**)error {
    NSURL* url = [self selectedAudioOutputURL];
    if (!url) {
        [self resetPlaybackProgress];
        return NO;
    }
    if (!self.audioPlayer || ![self.audioPlayer.url isEqual:url]) {
        [self stopPlaybackTimer];
        self.audioPlayer = [[AVAudioPlayer alloc] initWithContentsOfURL:url error:error];
        if (!self.audioPlayer) {
            [self resetPlaybackProgress];
            return NO;
        }
        self.audioPlayer.delegate = self;
        [self.audioPlayer prepareToPlay];
    }
    if (startTime > 0.0 && startTime < self.audioPlayer.duration) {
        self.audioPlayer.currentTime = startTime;
    } else if (startTime <= 0.0) {
        self.audioPlayer.currentTime = 0.0;
    }
    [self updatePlaybackProgress];
    return YES;
}

- (void)stopPlayback {
    [self stopPlaybackTimer];
    [self.audioPlayer stop];
    self.audioPlayer = nil;
    self.playbackScrubbing = NO;
    [self setButton:self.playButton title:@"播放"];
    [self resetPlaybackProgress];
}

- (void)refreshOutputControls {
    [self.outputSelector removeAllSegments];
    for (NSUInteger i = 0; i < self.audioOutputNames.count; ++i) {
        [self.outputSelector insertSegmentWithTitle:self.audioOutputNames[i] atIndex:i animated:NO];
    }
    const BOOL hasAudio = self.audioOutputURLs.count > 0;
    self.outputSelector.hidden = !hasAudio;
    if (hasAudio) {
        self.outputSelector.selectedSegmentIndex = 0;
        NSString* name = [self selectedAudioOutputName] ?: @"音轨";
        self.outputLabel.text = [NSString stringWithFormat:@"%@ 已生成", name];
    } else if (self.running) {
        self.outputLabel.text = @"正在分离音频";
    } else {
        self.outputLabel.text = self.emptyOutputMessage ?: @"还没有输出音轨";
    }
    self.playButton.enabled = !self.running && hasAudio;
    self.shareButton.enabled = !self.running && hasAudio;
    if (!hasAudio) {
        [self resetPlaybackProgress];
    } else if (!self.running && !self.audioPlayer) {
        NSError* error = nil;
        if (![self prepareSelectedAudioPlayerWithStartTime:0.0 error:&error]) {
            [self appendLog:[NSString stringWithFormat:@"load player failed: %@", error.localizedDescription ?: @"unknown error"]];
        }
    }
}

- (void)resetOutputsWithMessage:(NSString*)message {
    [self.outputURLs removeAllObjects];
    [self.audioOutputURLs removeAllObjects];
    [self.audioOutputNames removeAllObjects];
    self.emptyOutputMessage = message;
    [self refreshOutputControls];
}

- (void)outputSelectionChanged {
    [self stopPlayback];
    NSString* name = [self selectedAudioOutputName] ?: @"音轨";
    self.outputLabel.text = [NSString stringWithFormat:@"%@ 已生成", name];
    NSError* error = nil;
    if (![self prepareSelectedAudioPlayerWithStartTime:0.0 error:&error]) {
        [self appendLog:[NSString stringWithFormat:@"load player failed: %@", error.localizedDescription ?: @"unknown error"]];
    }
}

- (void)playbackSliderTouchDown {
    if (!self.audioPlayer) {
        return;
    }
    self.playbackScrubbing = YES;
}

- (void)playbackSliderValueChanged {
    if (!self.audioPlayer) {
        return;
    }
    [self setPlaybackPosition:self.playbackSlider.value duration:self.audioPlayer.duration];
}

- (void)playbackSliderTouchUp {
    if (!self.audioPlayer) {
        self.playbackScrubbing = NO;
        return;
    }
    self.audioPlayer.currentTime = std::max(0.0, std::min(static_cast<double>(self.playbackSlider.value), self.audioPlayer.duration));
    self.playbackScrubbing = NO;
    [self updatePlaybackProgress];
}

- (void)diagnosticsButtonTapped {
    self.diagnosticsVisible = !self.diagnosticsVisible;
    self.logView.hidden = !self.diagnosticsVisible;
    [self setButton:self.diagnosticsButton title:self.diagnosticsVisible ? @"隐藏诊断信息" : @"诊断信息"];
}

- (void)selectButtonTapped {
    if (self.running || self.window.rootViewController.presentedViewController != nil) {
        return;
    }
    UIDocumentPickerViewController* picker = [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:@[UTTypeAudio]
                                                                                                         asCopy:NO];
    NSURL* directoryURL = LastPickerDirectoryURL();
    if (directoryURL) {
        picker.directoryURL = directoryURL;
    }
    picker.delegate = self;
    picker.allowsMultipleSelection = NO;
    [self.window.rootViewController presentViewController:picker animated:YES completion:nil];
}

- (void)inputPlayButtonTapped {
    if (self.running) {
        return;
    }
    if (self.inputAudioPlayer.isPlaying) {
        [self.inputAudioPlayer pause];
        [self setButton:self.inputPlayButton title:@"试听输入"];
        [self stopInputPlaybackTimer];
        [self updateInputPlaybackProgress];
        return;
    }

    [self stopPlayback];
    NSError* error = nil;
    const NSTimeInterval requestedTime = static_cast<NSTimeInterval>(self.inputPlaybackSlider.value);
    if (![self prepareInputAudioPlayerWithStartTime:requestedTime error:&error]) {
        [self styleStatusForText:@"Failed"];
        [self appendLog:[NSString stringWithFormat:@"input play failed: %@", error.localizedDescription ?: @"unknown error"]];
        return;
    }
    if (self.inputAudioPlayer.duration > 0.0 && self.inputAudioPlayer.currentTime >= self.inputAudioPlayer.duration - 0.05) {
        self.inputAudioPlayer.currentTime = 0.0;
    }
    if ([self.inputAudioPlayer play]) {
        [self setButton:self.inputPlayButton title:@"暂停输入"];
        [self updateInputPlaybackProgress];
        [self startInputPlaybackTimer];
        [self appendLog:[NSString stringWithFormat:@"playing input: %@", self.inputAudioPlayer.url.lastPathComponent]];
    } else {
        [self styleStatusForText:@"Failed"];
        [self appendLog:@"input play failed"];
    }
}

- (void)runButtonTapped {
    [self runSmokeTest];
}

- (void)playButtonTapped {
    if (self.running || self.audioOutputURLs.count == 0) {
        return;
    }
    if (self.audioPlayer.isPlaying) {
        [self.audioPlayer pause];
        [self setButton:self.playButton title:@"播放"];
        [self stopPlaybackTimer];
        [self updatePlaybackProgress];
        return;
    }

    NSURL* url = [self selectedAudioOutputURL];
    if (!url) {
        return;
    }
    [self stopInputPlayback];
    if (!self.audioPlayer || ![self.audioPlayer.url isEqual:url]) {
        NSError* error = nil;
        const NSTimeInterval requestedTime = static_cast<NSTimeInterval>(self.playbackSlider.value);
        if (![self prepareSelectedAudioPlayerWithStartTime:requestedTime error:&error]) {
            [self styleStatusForText:@"Failed"];
            [self appendLog:[NSString stringWithFormat:@"play failed: %@", error.localizedDescription ?: @"unknown error"]];
            return;
        }
    }
    if (self.audioPlayer.duration > 0.0 && self.audioPlayer.currentTime >= self.audioPlayer.duration - 0.05) {
        self.audioPlayer.currentTime = 0.0;
    }
    if ([self.audioPlayer play]) {
        [self setButton:self.playButton title:@"暂停"];
        [self updatePlaybackProgress];
        [self startPlaybackTimer];
        [self appendLog:[NSString stringWithFormat:@"playing: %@", url.lastPathComponent]];
    } else {
        [self styleStatusForText:@"Failed"];
        [self appendLog:@"play failed"];
    }
}

- (void)shareButtonTapped {
    NSURL* url = [self selectedAudioOutputURL];
    if (!url || self.window.rootViewController.presentedViewController != nil) {
        return;
    }
    UIActivityViewController* controller = [[UIActivityViewController alloc] initWithActivityItems:@[url]
                                                                            applicationActivities:nil];
    controller.popoverPresentationController.sourceView = self.shareButton;
    controller.popoverPresentationController.sourceRect = self.shareButton.bounds;
    [self.window.rootViewController presentViewController:controller animated:YES completion:nil];
}

- (void)documentPicker:(UIDocumentPickerViewController*)controller didPickDocumentsAtURLs:(NSArray<NSURL*>*)urls {
    NSURL* sourceURL = urls.firstObject;
    if (!sourceURL) {
        return;
    }
    BOOL scoped = [sourceURL startAccessingSecurityScopedResource];
    @try {
        SaveLastPickerDirectoryURL(sourceURL.URLByDeletingLastPathComponent);
        NSString* extension = sourceURL.pathExtension.length > 0 ? sourceURL.pathExtension : @"audio";
        NSString* destinationName = [NSString stringWithFormat:@"selected_input.%@", extension];
        NSURL* destinationURL = [NSURL fileURLWithPath:DocumentsPath(destinationName)];
        NSFileManager* fileManager = [NSFileManager defaultManager];
        NSError* error = nil;
        [fileManager removeItemAtURL:destinationURL error:nil];
        if (![fileManager copyItemAtURL:sourceURL toURL:destinationURL error:&error]) {
            [self styleStatusForText:@"Failed"];
            [self appendLog:[NSString stringWithFormat:@"select failed: %@", error.localizedDescription ?: @"unknown error"]];
            return;
        }
        [self stopInputPlayback];
        [self stopPlayback];
        self.selectedInputURL = destinationURL;
        self.inputLabel.text = sourceURL.lastPathComponent;
        if (![self prepareInputAudioPlayerWithStartTime:0.0 error:&error]) {
            [self appendLog:[NSString stringWithFormat:@"load input player failed: %@", error.localizedDescription ?: @"unknown error"]];
        }
        self.metricsLabel.text = @"就绪\n初始化 --   推理 --\n实时率 --    速度 --";
        [self setProgressValue:0.0f title:@"等待开始"];
        [self resetOutputsWithMessage:@"开始分离后会在这里显示音轨"];
        [self styleStatusForText:@"Ready"];
        [self appendLog:[NSString stringWithFormat:@"selected: %@", sourceURL.lastPathComponent]];
    } @finally {
        if (scoped) {
            [sourceURL stopAccessingSecurityScopedResource];
        }
    }
}

- (void)documentPickerWasCancelled:(UIDocumentPickerViewController*)controller {
    [self appendLog:@"selection cancelled"];
}

- (void)audioPlayerDidFinishPlaying:(AVAudioPlayer*)player successfully:(BOOL)flag {
    if (player == self.inputAudioPlayer) {
        [self stopInputPlaybackTimer];
        [self updateInputPlaybackProgress];
        [self setButton:self.inputPlayButton title:@"试听输入"];
        return;
    }
    if (player == self.audioPlayer) {
        [self stopPlaybackTimer];
        [self updatePlaybackProgress];
        [self setButton:self.playButton title:@"播放"];
    }
}

- (void)setRunState:(BOOL)running status:(NSString*)status {
    dispatch_async(dispatch_get_main_queue(), ^{
        self.running = running;
        [self styleStatusForText:status];
        self.selectButton.enabled = !running;
        self.inputPlayButton.enabled = !running;
        self.runButton.enabled = !running;
        self.inputPlaybackSlider.enabled = !running && self.inputAudioPlayer.duration > 0.0;
        [self refreshOutputControls];
        if (running) {
            [self.activityView startAnimating];
            self.peakResidentBytes = CurrentResidentBytes();
            [self updateMonitor];
            [self startMonitorTimer];
        } else {
            [self.activityView stopAnimating];
            [self stopMonitorTimer];
            [self updateMonitor];
            if (!self.inputAudioPlayer) {
                NSError* inputError = nil;
                if (![self prepareInputAudioPlayerWithStartTime:0.0 error:&inputError]) {
                    [self appendLog:[NSString stringWithFormat:@"load input player failed: %@",
                                                                inputError.localizedDescription ?: @"unknown error"]];
                }
            }
        }
    });
}

- (void)startMonitorTimer {
    if (self.monitorTimer) {
        return;
    }
    self.monitorTimer = [NSTimer scheduledTimerWithTimeInterval:1.0
                                                         target:self
                                                       selector:@selector(updateMonitor)
                                                       userInfo:nil
                                                        repeats:YES];
    self.monitorTimer.tolerance = 0.2;
}

- (void)stopMonitorTimer {
    [self.monitorTimer invalidate];
    self.monitorTimer = nil;
}

- (void)updateMonitor {
    const double cpu = CurrentProcessCPUPercent();
    const uint64_t resident = CurrentResidentBytes();
    if (resident > self.peakResidentBytes) {
        self.peakResidentBytes = resident;
    }

    NSString* cpuText = cpu >= 0.0 ? [NSString stringWithFormat:@"%.0f%%", cpu] : @"--";
    NSString* residentText = resident > 0 ? FormatBytes(resident) : @"--";
    NSString* peakText = self.peakResidentBytes > 0 ? FormatBytes(self.peakResidentBytes) : @"--";
    self.monitorLabel.text = [NSString stringWithFormat:@"CPU %@\n内存 %@  峰值 %@", cpuText, residentText, peakText];
}

- (void)setMetricsText:(NSString*)text {
    dispatch_async(dispatch_get_main_queue(), ^{
        self.metricsLabel.text = text;
    });
}

- (void)setProgressValue:(float)value title:(NSString*)title {
    dispatch_async(dispatch_get_main_queue(), ^{
        const float clamped = std::max(0.0f, std::min(1.0f, value));
        const int percent = static_cast<int>(std::round(clamped * 100.0f));
        self.progressLabel.text = [NSString stringWithFormat:@"%@  %d%%", title ?: @"处理中", percent];
        [self.progressView setProgress:clamped animated:self.running];
    });
}

- (void)appendLog:(NSString*)message {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSString* line = [message stringByAppendingString:@"\n"];
        self.logView.text = [self.logView.text stringByAppendingString:line];
        if (self.logView.text.length > 0) {
            NSRange bottom = NSMakeRange(self.logView.text.length - 1, 1);
            [self.logView scrollRangeToVisible:bottom];
        }
        NSLog(@"%@", message);
    });
}

- (void)runSmokeTest {
    if (self.running) {
        return;
    }
    [self stopInputPlayback];
    [self stopPlayback];
    [self.outputURLs removeAllObjects];
    [self.audioOutputURLs removeAllObjects];
    [self.audioOutputNames removeAllObjects];
    [self refreshOutputControls];
    NSURL* selectedInputURL = self.selectedInputURL;
    self.currentRunID = [NSString stringWithFormat:@"%.3f", NSDate.date.timeIntervalSince1970];
    [self setRunState:YES status:@"Running"];
    [self setMetricsText:@"正在分离\n初始化 --   推理 --\n实时率 --    速度 --"];
    [self setProgressValue:0.0f title:@"准备分离"];
    dispatch_async(dispatch_get_main_queue(), ^{
        self.logView.text = @"";
    });
    [self appendLog:[NSString stringWithFormat:@"starting run_id=%@", self.currentRunID]];
    WriteProgress("running: starting run_id=" + ToString(self.currentRunID));
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        try {
            NSString* corePath = OptionalBundlePath(MSS_MNN_IOS_CORE_BUNDLE_NAME);
            NSString* segmentPath = OptionalBundlePath(MSS_MNN_IOS_SEGMENT_BUNDLE_NAME);
            NSString* metadataPath = BundlePath(MSS_MNN_IOS_METADATA_BUNDLE_NAME);
            NSURL* inputURL = selectedInputURL;
            if (!inputURL) {
                inputURL = [NSURL fileURLWithPath:BundlePath(MSS_MNN_IOS_INPUT_BUNDLE_NAME)];
            }

            if (segmentPath.length > 0) {
                [self appendLog:[NSString stringWithFormat:@"segments: %@", segmentPath.lastPathComponent]];
            } else if (corePath.length > 0) {
                [self appendLog:[NSString stringWithFormat:@"core: %@", corePath.lastPathComponent]];
            } else {
                throw std::runtime_error("missing bundled segment directory or core model");
            }
            [self appendLog:[NSString stringWithFormat:@"metadata: %@", metadataPath.lastPathComponent]];
            [self appendLog:[NSString stringWithFormat:@"input: %@", inputURL.lastPathComponent]];
            WriteProgress("running: resources ready");
            [self setProgressValue:0.03f title:@"准备资源"];

            mss_mnn::RoformerSeparatorOptions options;
            if (segmentPath.length > 0) {
                options.segment_dir = ToString(segmentPath);
            } else {
                options.core_model_path = ToString(corePath);
            }
            options.metadata_path = ToString(metadataPath);
            options.backend = mss_mnn::MNNBackend::Metal;
            options.precision = segmentPath.length > 0 ? mss_mnn::MNNPrecision::Auto : mss_mnn::MNNPrecision::Normal;
            options.precision_policy = mss_mnn::RoformerPrecisionPolicy::MetalFast;
            options.segment_cache_policy = mss_mnn::RoformerSegmentCachePolicy::Auto;
            options.attention_kernel = segmentPath.length > 0 ? mss_mnn::RoformerAttentionKernel::Simple
                                                              : mss_mnn::RoformerAttentionKernel::Fused;
            options.threads = 1;
            options.profile = true;
            options.profile_ops = true;
            options.profile_op_top_n = 30;
            options.progress_callback = [self](float value, const std::string&) {
                const float mapped = 0.20f + std::max(0.0f, std::min(1.0f, value)) * 0.72f;
                [self setProgressValue:mapped title:@"正在推理"];
            };

            const auto totalStart = Clock::now();
            const auto initStart = Clock::now();
            WriteProgress("running: initializing model");
            [self setProgressValue:0.06f title:@"加载模型"];
            mss_mnn::RoformerSeparator separator(options);
            const double initSeconds = ElapsedSeconds(initStart);
            [self setProgressValue:0.14f title:@"模型已加载"];
            {
                std::ostringstream progress;
                progress << std::fixed << std::setprecision(2) << "running: model initialized init=" << initSeconds << "s";
                WriteProgress(progress.str());
            }

            const auto readStart = Clock::now();
            WriteProgress("running: reading audio");
            [self setProgressValue:0.16f title:@"读取音频"];
            mss_mnn::AudioBuffer input = ReadAudioURL(inputURL, separator.metadata().sample_rate, 2);
            const double readSeconds = ElapsedSeconds(readStart);
            const double audioSeconds = input.sample_rate > 0 ? static_cast<double>(input.frames()) / input.sample_rate : 0.0;
            [self setProgressValue:0.20f title:@"音频已读取"];
            [self appendLog:[NSString stringWithFormat:@"audio: %.2fs, %d Hz, %d ch", audioSeconds, input.sample_rate, input.channels]];
            {
                std::ostringstream progress;
                progress << std::fixed << std::setprecision(2) << "running: audio ready audio=" << audioSeconds
                         << "s read=" << readSeconds << "s";
                WriteProgress(progress.str());
            }

            const auto inferStart = Clock::now();
            WriteProgress("running: inference");
            [self setProgressValue:0.20f title:@"正在推理"];
            std::vector<mss_mnn::AudioBuffer> outputs = separator.separate(input);
            const double inferSeconds = ElapsedSeconds(inferStart);
            const double rtf = audioSeconds > 0.0 ? inferSeconds / audioSeconds : 0.0;
            const double speed = inferSeconds > 0.0 ? audioSeconds / inferSeconds : 0.0;
            [self setProgressValue:0.92f title:@"推理完成"];
            [self setMetricsText:[NSString stringWithFormat:@"初始化 %.2fs  推理 %.2fs\n实时率 %.2f     速度 %.2fx",
                                                             initSeconds,
                                                             inferSeconds,
                                                             rtf,
                                                             speed]];
            {
                std::ostringstream progress;
                progress << std::fixed << std::setprecision(2) << "running: inference done infer=" << inferSeconds
                         << "s rtf=" << rtf;
                WriteProgress(progress.str());
            }

            const auto writeStart = Clock::now();
            WriteProgress("running: writing outputs");
            [self setProgressValue:0.95f title:@"写出音轨"];
            const auto& metadata = separator.metadata();
            NSMutableArray<NSURL*>* outputURLs = [NSMutableArray arrayWithObject:[NSURL fileURLWithPath:DocumentsPath(@"summary.txt")]];
            NSMutableArray<NSURL*>* audioOutputURLs = [NSMutableArray array];
            NSMutableArray<NSString*>* audioOutputNames = [NSMutableArray array];
            std::size_t vocalOutputIndex = outputs.size();
            bool hasInstrumentalOutput = false;
            for (std::size_t i = 0; i < outputs.size(); ++i) {
                const std::string stem = i < metadata.source_names.size() ? metadata.source_names[i] : std::to_string(i);
                if (IsVocalStemName(stem)) {
                    vocalOutputIndex = i;
                }
                if (stem == "instrumental" || stem == "accompaniment") {
                    hasInstrumentalOutput = true;
                }
                NSString* outputName = [NSString stringWithFormat:@"%s.wav", stem.c_str()];
                NSString* outputPath = DocumentsPath(outputName);
                mss_mnn::write_wav_float32(ToString(outputPath), outputs[i]);
                NSURL* outputURL = [NSURL fileURLWithPath:outputPath];
                [outputURLs addObject:outputURL];
                [audioOutputURLs addObject:outputURL];
                NSString* displayName = DisplayNameForStem([NSString stringWithUTF8String:stem.c_str()]);
                [audioOutputNames addObject:displayName.length > 0 ? displayName : outputName];
            }
            if (outputs.size() == 1 && vocalOutputIndex < outputs.size() && !hasInstrumentalOutput) {
                mss_mnn::AudioBuffer instrumental = MakeResidualAudio(input, outputs[vocalOutputIndex]);
                NSString* outputPath = DocumentsPath(@"instrumental.wav");
                mss_mnn::write_wav_float32(ToString(outputPath), instrumental);
                NSURL* outputURL = [NSURL fileURLWithPath:outputPath];
                [outputURLs addObject:outputURL];
                [audioOutputURLs addObject:outputURL];
                [audioOutputNames addObject:@"伴奏"];
            }
            const double writeSeconds = ElapsedSeconds(writeStart);
            const double totalSeconds = ElapsedSeconds(totalStart);

            std::ostringstream summary;
            summary << std::fixed << std::setprecision(2)
                    << "run_id=" << ToString(self.currentRunID)
                    << " done: init=" << initSeconds << "s read=" << readSeconds
                    << "s infer=" << inferSeconds << "s write=" << writeSeconds
                    << "s total=" << totalSeconds
                    << "s rtf=" << rtf << " speed=" << speed << "x";
            WriteTextFile(DocumentsPath(@"summary.txt"), summary.str() + "\n" + separator.last_profile_report());
            [self appendLog:[NSString stringWithUTF8String:summary.str().c_str()]];
            [self appendLog:@"outputs written to app Documents"];
            dispatch_async(dispatch_get_main_queue(), ^{
                self.outputURLs = outputURLs;
                self.audioOutputURLs = audioOutputURLs;
                self.audioOutputNames = audioOutputNames;
                [self refreshOutputControls];
            });
            [self setProgressValue:1.0f title:@"分离完成"];
            [self setRunState:NO status:@"Complete"];
            if (self.exitAfterRun) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    std::exit(0);
                });
            }
        } catch (const NSException* exception) {
            NSString* message = [NSString stringWithFormat:@"objc error: %@ %@", exception.name, exception.reason];
            WriteTextFile(DocumentsPath(@"summary.txt"), ToString(message) + "\n");
            [self appendLog:message];
            dispatch_async(dispatch_get_main_queue(), ^{
                self.outputURLs = [NSMutableArray arrayWithObject:[NSURL fileURLWithPath:DocumentsPath(@"summary.txt")]];
                self.audioOutputURLs = [NSMutableArray array];
                self.audioOutputNames = [NSMutableArray array];
                [self refreshOutputControls];
            });
            [self setMetricsText:@"运行失败"];
            [self setProgressValue:0.0f title:@"分离失败"];
            [self setRunState:NO status:@"Failed"];
            if (self.exitAfterRun) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    std::exit(1);
                });
            }
        } catch (const std::exception& error) {
            std::string message = std::string("c++ error: ") + error.what();
            WriteTextFile(DocumentsPath(@"summary.txt"), message + "\n");
            [self appendLog:[NSString stringWithUTF8String:message.c_str()]];
            dispatch_async(dispatch_get_main_queue(), ^{
                self.outputURLs = [NSMutableArray arrayWithObject:[NSURL fileURLWithPath:DocumentsPath(@"summary.txt")]];
                self.audioOutputURLs = [NSMutableArray array];
                self.audioOutputNames = [NSMutableArray array];
                [self refreshOutputControls];
            });
            [self setMetricsText:@"运行失败"];
            [self setProgressValue:0.0f title:@"分离失败"];
            [self setRunState:NO status:@"Failed"];
            if (self.exitAfterRun) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    std::exit(1);
                });
            }
        } catch (...) {
            WriteTextFile(DocumentsPath(@"summary.txt"), "unknown error\n");
            [self appendLog:@"unknown error"];
            dispatch_async(dispatch_get_main_queue(), ^{
                self.outputURLs = [NSMutableArray arrayWithObject:[NSURL fileURLWithPath:DocumentsPath(@"summary.txt")]];
                self.audioOutputURLs = [NSMutableArray array];
                self.audioOutputNames = [NSMutableArray array];
                [self refreshOutputControls];
            });
            [self setMetricsText:@"运行失败"];
            [self setProgressValue:0.0f title:@"分离失败"];
            [self setRunState:NO status:@"Failed"];
            if (self.exitAfterRun) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    std::exit(1);
                });
            }
        }
    });
}

@end

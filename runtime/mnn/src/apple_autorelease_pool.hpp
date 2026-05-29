#pragma once

#if defined(__APPLE__)
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void* pool);
#endif

namespace mss_mnn {

class ScopedAutoreleasePool {
public:
    ScopedAutoreleasePool() {
#if defined(__APPLE__)
        pool_ = objc_autoreleasePoolPush();
#endif
    }

    ~ScopedAutoreleasePool() {
#if defined(__APPLE__)
        objc_autoreleasePoolPop(pool_);
#endif
    }

    ScopedAutoreleasePool(const ScopedAutoreleasePool&) = delete;
    ScopedAutoreleasePool& operator=(const ScopedAutoreleasePool&) = delete;

private:
#if defined(__APPLE__)
    void* pool_ = nullptr;
#endif
};

}  // namespace mss_mnn

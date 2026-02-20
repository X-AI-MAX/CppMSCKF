#ifndef MSCKF_SYSTEM_SCHEDULER_HPP
#define MSCKF_SYSTEM_SCHEDULER_HPP

#include "../math/types.hpp"
#include "lockfree_queue.hpp"
#include "memory_pool.hpp"

namespace msckf {

enum class TaskPriority : uint8 {
    CRITICAL = 0,
    HIGH = 1,
    NORMAL = 2,
    LOW = 3,
    IDLE = 4
};

enum class TaskState : uint8 {
    PENDING = 0,
    RUNNING = 1,
    SUSPENDED = 2,
    COMPLETED = 3,
    ERROR = 4
};

typedef void (*TaskFunc)(void* context);

struct Task {
    TaskFunc func;
    void* context;
    TaskPriority priority;
    TaskState state;
    uint32 id;
    uint64 deadline;
    uint64 period;
    uint32 executionCount;
    uint64 totalExecutionTime;
    uint64 maxExecutionTime;
    uint64 lastExecutionTime;
    const char* name;
    
    Task() 
        : func(nullptr)
        , context(nullptr)
        , priority(TaskPriority::NORMAL)
        , state(TaskState::PENDING)
        , id(0)
        , deadline(0)
        , period(0)
        , executionCount(0)
        , totalExecutionTime(0)
        , maxExecutionTime(0)
        , lastExecutionTime(0)
        , name(nullptr) {}
};

class Scheduler {
public:
    static constexpr uint32 MAX_TASKS = 32;
    static constexpr uint32 MAX_PRIORITY_LEVELS = 5;

private:
    Task tasks_[MAX_TASKS];
    uint32 taskCount_;
    uint32 nextTaskId_;
    bool running_;
    
    Task* readyLists_[MAX_PRIORITY_LEVELS];
    Task* currentTask_;
    
    uint64 tickCount_;
    uint64 tickPeriodUs_;
    
    SPSCQueue<uint32, 64> eventQueue_;
    
    uint32 imuTaskId_;
    uint32 visualTaskId_;
    uint32 outputTaskId_;

public:
    Scheduler() 
        : taskCount_(0)
        , nextTaskId_(1)
        , running_(false)
        , currentTask_(nullptr)
        , tickCount_(0)
        , tickPeriodUs_(1000)
        , imuTaskId_(0)
        , visualTaskId_(0)
        , outputTaskId_(0) {
        for (uint32 i = 0; i < MAX_PRIORITY_LEVELS; ++i) {
            readyLists_[i] = nullptr;
        }
    }
    
    uint32 createTask(TaskFunc func, void* context, TaskPriority priority,
                      const char* name = nullptr, uint64 period = 0) {
        if (taskCount_ >= MAX_TASKS) {
            return 0;
        }
        
        uint32 slot = findFreeSlot();
        if (slot >= MAX_TASKS) {
            return 0;
        }
        
        Task& task = tasks_[slot];
        task.func = func;
        task.context = context;
        task.priority = priority;
        task.state = TaskState::PENDING;
        task.id = nextTaskId_++;
        task.period = period;
        task.deadline = 0;
        task.executionCount = 0;
        task.totalExecutionTime = 0;
        task.maxExecutionTime = 0;
        task.lastExecutionTime = 0;
        task.name = name;
        
        taskCount_++;
        
        addToReadyList(&task);
        
        return task.id;
    }
    
    void deleteTask(uint32 taskId) {
        for (uint32 i = 0; i < MAX_TASKS; ++i) {
            if (tasks_[i].id == taskId) {
                removeFromReadyList(&tasks_[i]);
                tasks_[i].state = TaskState::COMPLETED;
                tasks_[i].func = nullptr;
                taskCount_--;
                break;
            }
        }
    }
    
    void setImuTask(uint32 taskId) { imuTaskId_ = taskId; }
    void setVisualTask(uint32 taskId) { visualTaskId_ = taskId; }
    void setOutputTask(uint32 taskId) { outputTaskId_ = taskId; }
    
    void start() {
        running_ = true;
        run();
    }
    
    void stop() {
        running_ = false;
    }
    
    void tick() {
        tickCount_++;
        
        for (uint32 i = 0; i < MAX_TASKS; ++i) {
            Task& task = tasks_[i];
            if (task.func && task.period > 0) {
                if (tickCount_ % (task.period / tickPeriodUs_) == 0) {
                    task.state = TaskState::PENDING;
                    addToReadyList(&task);
                }
            }
        }
    }
    
    void run() {
        while (running_) {
            Task* task = selectNextTask();
            
            if (task) {
                executeTask(task);
            } else {
                idle();
            }
        }
    }
    
    void runOnce() {
        Task* task = selectNextTask();
        if (task) {
            executeTask(task);
        }
    }
    
    void suspendTask(uint32 taskId) {
        Task* task = findTask(taskId);
        if (task && task->state == TaskState::RUNNING) {
            task->state = TaskState::SUSPENDED;
            removeFromReadyList(task);
        }
    }
    
    void resumeTask(uint32 taskId) {
        Task* task = findTask(taskId);
        if (task && task->state == TaskState::SUSPENDED) {
            task->state = TaskState::PENDING;
            addToReadyList(task);
        }
    }
    
    void postEvent(uint32 eventId) {
        eventQueue_.push(eventId);
    }
    
    bool getEvent(uint32& eventId) {
        return eventQueue_.pop(eventId);
    }
    
    uint64 getTickCount() const { return tickCount_; }
    uint64 getTimeUs() const { return tickCount_ * tickPeriodUs_; }
    
    void setTickPeriod(uint64 periodUs) { tickPeriodUs_ = periodUs; }
    
    const Task* getTask(uint32 taskId) const {
        return findTask(taskId);
    }
    
    uint32 getTaskCount() const { return taskCount_; }

private:
    uint32 findFreeSlot() const {
        for (uint32 i = 0; i < MAX_TASKS; ++i) {
            if (tasks_[i].func == nullptr) {
                return i;
            }
        }
        return MAX_TASKS;
    }
    
    Task* findTask(uint32 taskId) {
        for (uint32 i = 0; i < MAX_TASKS; ++i) {
            if (tasks_[i].id == taskId) {
                return &tasks_[i];
            }
        }
        return nullptr;
    }
    
    void addToReadyList(Task* task) {
        uint32 prio = static_cast<uint32>(task->priority);
        task->state = TaskState::PENDING;
    }
    
    void removeFromReadyList(Task* task) {
    }
    
    Task* selectNextTask() {
        for (uint32 prio = 0; prio < MAX_PRIORITY_LEVELS; ++prio) {
            for (uint32 i = 0; i < MAX_TASKS; ++i) {
                Task& task = tasks_[i];
                if (task.func && task.state == TaskState::PENDING && 
                    static_cast<uint32>(task.priority) == prio) {
                    return &task;
                }
            }
        }
        return nullptr;
    }
    
    void executeTask(Task* task) {
        currentTask_ = task;
        task->state = TaskState::RUNNING;
        
        uint64 startTime = getTimeUs();
        
        if (task->func) {
            task->func(task->context);
        }
        
        uint64 endTime = getTimeUs();
        uint64 execTime = endTime - startTime;
        
        task->lastExecutionTime = execTime;
        task->totalExecutionTime += execTime;
        task->executionCount++;
        
        if (execTime > task->maxExecutionTime) {
            task->maxExecutionTime = execTime;
        }
        
        if (task->period == 0) {
            task->state = TaskState::COMPLETED;
            removeFromReadyList(task);
        } else {
            task->state = TaskState::PENDING;
        }
        
        currentTask_ = nullptr;
    }
    
    void idle() {
        asm volatile("wfi" ::: "memory");
    }
};

class Timer {
private:
    uint64 startTime_;
    uint64 period_;
    bool running_;
    bool periodic_;
    void (*callback_)(void*);
    void* context_;

public:
    Timer() 
        : startTime_(0)
        , period_(0)
        , running_(false)
        , periodic_(false)
        , callback_(nullptr)
        , context_(nullptr) {}
    
    void start(uint64 periodUs, void (*callback)(void*) = nullptr, void* ctx = nullptr) {
        startTime_ = 0;
        period_ = periodUs;
        callback_ = callback;
        context_ = ctx;
        running_ = true;
        periodic_ = false;
    }
    
    void startPeriodic(uint64 periodUs, void (*callback)(void*) = nullptr, void* ctx = nullptr) {
        startTime_ = 0;
        period_ = periodUs;
        callback_ = callback;
        context_ = ctx;
        running_ = true;
        periodic_ = true;
    }
    
    void stop() {
        running_ = false;
    }
    
    bool check(uint64 currentTime) {
        if (!running_) return false;
        
        if (currentTime - startTime_ >= period_) {
            if (callback_) {
                callback_(context_);
            }
            
            if (periodic_) {
                startTime_ = currentTime;
            } else {
                running_ = false;
            }
            return true;
        }
        return false;
    }
    
    bool isRunning() const { return running_; }
    uint64 getPeriod() const { return period_; }
};

class Watchdog {
private:
    uint64 timeout_;
    uint64 lastFeed_;
    bool enabled_;

public:
    Watchdog() 
        : timeout_(1000000)
        , lastFeed_(0)
        , enabled_(false) {}
    
    void init(uint64 timeoutUs) {
        timeout_ = timeoutUs;
        lastFeed_ = 0;
        enabled_ = true;
    }
    
    void feed(uint64 currentTime) {
        lastFeed_ = currentTime;
    }
    
    bool check(uint64 currentTime) {
        if (!enabled_) return true;
        return (currentTime - lastFeed_) < timeout_;
    }
    
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool isEnabled() const { return enabled_; }
};

}

#endif

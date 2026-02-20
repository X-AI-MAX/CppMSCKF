#ifndef MSCKF_AUDIT_SYSTEM_DOCUMENTATION_HPP
#define MSCKF_AUDIT_SYSTEM_DOCUMENTATION_HPP

#include "../math/types.hpp"

namespace msckf {

namespace doc {

struct DataFlowNode {
    const char* id;
    const char* label;
    const char* type;
    const char* description;
    
    DataFlowNode() 
        : id(nullptr), label(nullptr), type(nullptr), description(nullptr) {}
    
    DataFlowNode(const char* i, const char* l, const char* t, const char* d)
        : id(i), label(l), type(t), description(d) {}
};

struct DataFlowEdge {
    const char* sourceId;
    const char* targetId;
    const char* dataType;
    const char* description;
    float64 rate;
    
    DataFlowEdge() 
        : sourceId(nullptr), targetId(nullptr), dataType(nullptr)
        , description(nullptr), rate(0) {}
    
    DataFlowEdge(const char* s, const char* t, const char* dt, 
                const char* d, float64 r)
        : sourceId(s), targetId(t), dataType(dt), description(d), rate(r) {}
};

class DataFlowDiagram {
public:
    static constexpr uint32 MAX_NODES = 50;
    static constexpr uint32 MAX_EDGES = 100;

private:
    DataFlowNode nodes_[MAX_NODES];
    DataFlowEdge edges_[MAX_EDGES];
    uint32 nodeCount_;
    uint32 edgeCount_;
    
    const char* title_;
    const char* version_;

public:
    DataFlowDiagram() 
        : nodeCount_(0), edgeCount_(0)
        , title_("MSCKF Data Flow Diagram")
        , version_("1.0") {
        initializeDefaultNodes();
        initializeDefaultEdges();
    }
    
    void initializeDefaultNodes() {
        addNode("IMU_SENSOR", "IMU Sensor", "SENSOR", "Inertial Measurement Unit");
        addNode("CAMERA_SENSOR", "Camera Sensor", "SENSOR", "Image Sensor");
        addNode("IMU_HAL", "IMU HAL", "HAL", "IMU Hardware Abstraction");
        addNode("CAMERA_HAL", "Camera HAL", "HAL", "Camera Hardware Abstraction");
        addNode("IMU_QUEUE", "IMU Queue", "BUFFER", "Lock-free IMU data buffer");
        addNode("IMAGE_QUEUE", "Image Queue", "BUFFER", "Lock-free image buffer");
        addNode("IMU_PROPAGATOR", "IMU Propagator", "CORE", "State propagation engine");
        addNode("FEATURE_DETECTOR", "Feature Detector", "VISION", "Corner detection");
        addNode("KLT_TRACKER", "KLT Tracker", "VISION", "Feature tracking");
        addNode("IMAGE_PROCESSOR", "Image Processor", "VISION", "Undistortion & preprocessing");
        addNode("TRIANGULATION", "Triangulation", "CORE", "Feature triangulation");
        addNode("VISUAL_UPDATER", "Visual Updater", "CORE", "EKF visual update");
        addNode("MARGINALIZER", "Marginalizer", "CORE", "State marginalization");
        addNode("STATE_MANAGER", "State Manager", "CORE", "MSCKF state container");
        addNode("COVARIANCE", "Covariance Matrix", "DATA", "Error covariance P");
        addNode("OUTPUT_QUEUE", "Output Queue", "BUFFER", "Pose output buffer");
        addNode("COMM_INTERFACE", "Communication", "HAL", "External interface");
    }
    
    void initializeDefaultEdges() {
        addEdge("IMU_SENSOR", "IMU_HAL", "IMUData", "Raw IMU samples", 1000.0);
        addEdge("CAMERA_SENSOR", "CAMERA_HAL", "ImageData", "Raw image frames", 30.0);
        addEdge("IMU_HAL", "IMU_QUEUE", "IMUData", "Calibrated IMU data", 1000.0);
        addEdge("CAMERA_HAL", "IMAGE_QUEUE", "ImageData", "Raw Bayer images", 30.0);
        addEdge("IMU_QUEUE", "IMU_PROPAGATOR", "IMUData", "IMU measurements", 1000.0);
        addEdge("IMAGE_QUEUE", "IMAGE_PROCESSOR", "ImageData", "Image frames", 30.0);
        addEdge("IMAGE_PROCESSOR", "FEATURE_DETECTOR", "uint8*", "Undistorted gray", 30.0);
        addEdge("IMAGE_PROCESSOR", "KLT_TRACKER", "uint8*", "Undistorted gray", 30.0);
        addEdge("FEATURE_DETECTOR", "KLT_TRACKER", "Feature2D[]", "New features", 30.0);
        addEdge("KLT_TRACKER", "TRIANGULATION", "Feature[]", "Tracked features", 30.0);
        addEdge("TRIANGULATION", "VISUAL_UPDATER", "Feature[]", "Triangulated features", 30.0);
        addEdge("IMU_PROPAGATOR", "STATE_MANAGER", "IMUState", "Propagated state", 1000.0);
        addEdge("STATE_MANAGER", "VISUAL_UPDATER", "MSCKFState", "Current state", 30.0);
        addEdge("VISUAL_UPDATER", "STATE_MANAGER", "ErrorState", "State correction", 30.0);
        addEdge("VISUAL_UPDATER", "COVARIANCE", "Matrix", "Covariance update", 30.0);
        addEdge("STATE_MANAGER", "MARGINALIZER", "CameraState[]", "Oldest frame", 1.0);
        addEdge("MARGINALIZER", "COVARIANCE", "Matrix", "Marginalized P", 1.0);
        addEdge("STATE_MANAGER", "OUTPUT_QUEUE", "Pose", "Estimated pose", 100.0);
        addEdge("OUTPUT_QUEUE", "COMM_INTERFACE", "Pose", "Output pose", 100.0);
    }
    
    void addNode(const char* id, const char* label, const char* type, const char* desc) {
        if (nodeCount_ < MAX_NODES) {
            nodes_[nodeCount_++] = DataFlowNode(id, label, type, desc);
        }
    }
    
    void addEdge(const char* src, const char* tgt, const char* dataType, 
                const char* desc, float64 rate) {
        if (edgeCount_ < MAX_EDGES) {
            edges_[edgeCount_++] = DataFlowEdge(src, tgt, dataType, desc, rate);
        }
    }
    
    void generateMermaid(char* buffer, uint32 bufferSize) const {
        uint32 offset = 0;
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "```mermaid\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "flowchart TB\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    %% %s v%s\n\n", title_, version_);
        
        for (uint32 i = 0; i < nodeCount_; ++i) {
            const DataFlowNode& node = nodes_[i];
            const char* shape = "rect";
            
            if (strcmp(node.type, "SENSOR") == 0) shape = "cyl";
            else if (strcmp(node.type, "BUFFER") == 0) shape = "parallelogram";
            else if (strcmp(node.type, "DATA") == 0) shape = "stadium";
            
            offset += snprintf(buffer + offset, bufferSize - offset,
                "    %s%s\"%s\":::%s\n", 
                node.id, shape, node.label, node.type);
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset, "\n");
        
        for (uint32 i = 0; i < edgeCount_; ++i) {
            const DataFlowEdge& edge = edges_[i];
            offset += snprintf(buffer + offset, bufferSize - offset,
                "    %s -->|%s %.0fHz| %s\n",
                edge.sourceId, edge.dataType, edge.rate, edge.targetId);
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset, "\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    classDef SENSOR fill:#ff9999\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    classDef HAL fill:#99ff99\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    classDef BUFFER fill:#9999ff\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    classDef CORE fill:#ffff99\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    classDef VISION fill:#ff99ff\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    classDef DATA fill:#99ffff\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "```\n");
    }
    
    void generatePlantUML(char* buffer, uint32 bufferSize) const {
        uint32 offset = 0;
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "@startuml\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "!define RECTANGLE class\n\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "skinparam componentStyle rectangle\n\n");
        
        for (uint32 i = 0; i < nodeCount_; ++i) {
            const DataFlowNode& node = nodes_[i];
            offset += snprintf(buffer + offset, bufferSize - offset,
                "component \"%s\" as %s <<%s>>\n",
                node.label, node.id, node.type);
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset, "\n");
        
        for (uint32 i = 0; i < edgeCount_; ++i) {
            const DataFlowEdge& edge = edges_[i];
            offset += snprintf(buffer + offset, bufferSize - offset,
                "%s -> %s : %s\n",
                edge.sourceId, edge.targetId, edge.dataType);
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "@enduml\n");
    }
    
    uint32 getNodeCount() const { return nodeCount_; }
    uint32 getEdgeCount() const { return edgeCount_; }
    const DataFlowNode& getNode(uint32 idx) const { return nodes_[idx]; }
    const DataFlowEdge& getEdge(uint32 idx) const { return edges_[idx]; }
};

struct StateMachineState {
    const char* id;
    const char* label;
    const char* description;
    bool isInitial;
    bool isFinal;
    
    StateMachineState() 
        : id(nullptr), label(nullptr), description(nullptr)
        , isInitial(false), isFinal(false) {}
};

struct StateMachineTransition {
    const char* sourceId;
    const char* targetId;
    const char* trigger;
    const char* guard;
    const char* action;
    
    StateMachineTransition() 
        : sourceId(nullptr), targetId(nullptr), trigger(nullptr)
        , guard(nullptr), action(nullptr) {}
};

class StateMachineDiagram {
public:
    static constexpr uint32 MAX_STATES = 20;
    static constexpr uint32 MAX_TRANSITIONS = 50;

private:
    StateMachineState states_[MAX_STATES];
    StateMachineTransition transitions_[MAX_TRANSITIONS];
    uint32 stateCount_;
    uint32 transitionCount_;
    
    const char* title_;

public:
    StateMachineDiagram() 
        : stateCount_(0), transitionCount_(0)
        , title_("MSCKF System State Machine") {
        initializeDefaultStates();
        initializeDefaultTransitions();
    }
    
    void initializeDefaultStates() {
        addState("INIT", "Initialization", "System startup and calibration", true, false);
        addState("NORMAL", "Normal Operation", "Regular IMU propagation and visual update", false, false);
        addState("PURE_ROTATION", "Pure Rotation Mode", "Degenerate case with no translation", false, false);
        addState("IMU_ONLY", "IMU-Only Mode", "Visual lost, pure IMU propagation", false, false);
        addState("SHOCK_RECOVERY", "Shock Recovery", "Recovering from impact/shock", false, false);
        addState("LOW_POWER", "Low Power Mode", "Reduced update rate for power saving", false, false);
        addState("ERROR", "Error State", "Critical error, needs reset", false, true);
    }
    
    void initializeDefaultTransitions() {
        addTransition("INIT", "NORMAL", "initialization_complete", "calibrated", "startProcessing()");
        addTransition("NORMAL", "PURE_ROTATION", "translation_below_threshold", "||t|| < 0.05m", "fixInverseDepth()");
        addTransition("PURE_ROTATION", "NORMAL", "translation_above_threshold", "||t|| > 0.1m", "restoreFullState()");
        addTransition("NORMAL", "IMU_ONLY", "visual_lost", "no_features > 5s", "inflateCovariance()");
        addTransition("IMU_ONLY", "NORMAL", "visual_recovered", "features_detected", "deflateCovariance()");
        addTransition("NORMAL", "SHOCK_RECOVERY", "shock_detected", "||a|| > 10g", "savePreShockState()");
        addTransition("SHOCK_RECOVERY", "NORMAL", "recovery_complete", "stable > 1s", "restoreState()");
        addTransition("NORMAL", "LOW_POWER", "low_battery", "battery < 20%", "reduceUpdateRate()");
        addTransition("LOW_POWER", "NORMAL", "power_restored", "battery > 30%", "restoreUpdateRate()");
        addTransition("NORMAL", "ERROR", "critical_error", "unrecoverable", "logError()");
        addTransition("IMU_ONLY", "ERROR", "imu_failure", "imu_timeout > 1s", "logError()");
        addTransition("SHOCK_RECOVERY", "ERROR", "recovery_failed", "attempts > 3", "logError()");
    }
    
    void addState(const char* id, const char* label, const char* desc, 
                 bool isInitial, bool isFinal) {
        if (stateCount_ < MAX_STATES) {
            states_[stateCount_].id = id;
            states_[stateCount_].label = label;
            states_[stateCount_].description = desc;
            states_[stateCount_].isInitial = isInitial;
            states_[stateCount_].isFinal = isFinal;
            stateCount_++;
        }
    }
    
    void addTransition(const char* src, const char* tgt, const char* trigger,
                      const char* guard, const char* action) {
        if (transitionCount_ < MAX_TRANSITIONS) {
            transitions_[transitionCount_].sourceId = src;
            transitions_[transitionCount_].targetId = tgt;
            transitions_[transitionCount_].trigger = trigger;
            transitions_[transitionCount_].guard = guard;
            transitions_[transitionCount_].action = action;
            transitionCount_++;
        }
    }
    
    void generateMermaid(char* buffer, uint32 bufferSize) const {
        uint32 offset = 0;
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "```mermaid\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "stateDiagram-v2\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    %% %s\n\n", title_);
        
        for (uint32 i = 0; i < stateCount_; ++i) {
            const StateMachineState& state = states_[i];
            if (state.isInitial) {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "    [*] --> %s\n", state.id);
            }
            if (state.isFinal) {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "    %s --> [*]\n", state.id);
            }
            
            offset += snprintf(buffer + offset, bufferSize - offset,
                "    %s : %s\n", state.id, state.label);
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset, "\n");
        
        for (uint32 i = 0; i < transitionCount_; ++i) {
            const StateMachineTransition& t = transitions_[i];
            if (t.guard && t.guard[0] != '\0') {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "    %s --> %s : %s\\n[%s]\n",
                    t.sourceId, t.targetId, t.trigger, t.guard);
            } else {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "    %s --> %s : %s\n",
                    t.sourceId, t.targetId, t.trigger);
            }
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "```\n");
    }
    
    void generatePlantUML(char* buffer, uint32 bufferSize) const {
        uint32 offset = 0;
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "@startuml\n\n");
        
        for (uint32 i = 0; i < stateCount_; ++i) {
            const StateMachineState& state = states_[i];
            if (state.isInitial) {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "[*] --> %s\n", state.id);
            }
            if (state.isFinal) {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "%s --> [*]\n", state.id);
            }
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset, "\n");
        
        for (uint32 i = 0; i < stateCount_; ++i) {
            const StateMachineState& state = states_[i];
            offset += snprintf(buffer + offset, bufferSize - offset,
                "state \"%s\" as %s\n", state.label, state.id);
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset, "\n");
        
        for (uint32 i = 0; i < transitionCount_; ++i) {
            const StateMachineTransition& t = transitions_[i];
            if (t.guard && t.guard[0] != '\0') {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "%s --> %s : %s\\n[%s]\n",
                    t.sourceId, t.targetId, t.trigger, t.guard);
            } else {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "%s --> %s : %s\n",
                    t.sourceId, t.targetId, t.trigger);
            }
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "@enduml\n");
    }
    
    uint32 getStateCount() const { return stateCount_; }
    uint32 getTransitionCount() const { return transitionCount_; }
    const StateMachineState& getState(uint32 idx) const { return states_[idx]; }
    const StateMachineTransition& getTransition(uint32 idx) const { return transitions_[idx]; }
};

struct SequenceParticipant {
    const char* id;
    const char* name;
    const char* type;
    
    SequenceParticipant() : id(nullptr), name(nullptr), type(nullptr) {}
};

struct SequenceMessage {
    const char* fromId;
    const char* toId;
    const char* message;
    const char* returnType;
    bool isAsync;
    bool isReturn;
    
    SequenceMessage() 
        : fromId(nullptr), toId(nullptr), message(nullptr)
        , returnType(nullptr), isAsync(false), isReturn(false) {}
};

struct SequenceLoop {
    const char* condition;
    uint32 startMessage;
    uint32 endMessage;
};

class SequenceDiagram {
public:
    static constexpr uint32 MAX_PARTICIPANTS = 15;
    static constexpr uint32 MAX_MESSAGES = 50;

private:
    SequenceParticipant participants_[MAX_PARTICIPANTS];
    SequenceMessage messages_[MAX_MESSAGES];
    SequenceLoop loops_[10];
    
    uint32 participantCount_;
    uint32 messageCount_;
    uint32 loopCount_;
    
    const char* title_;

public:
    SequenceDiagram() 
        : participantCount_(0), messageCount_(0), loopCount_(0)
        , title_("MSCKF IMU/Visual Thread Interaction") {
        initializeDefaultParticipants();
        initializeDefaultMessages();
    }
    
    void initializeDefaultParticipants() {
        addParticipant("IMU_THREAD", "IMU Thread", "thread");
        addParticipant("VISUAL_THREAD", "Visual Thread", "thread");
        addParticipant("IMU_QUEUE", "IMU Queue", "buffer");
        addParticipant("IMAGE_QUEUE", "Image Queue", "buffer");
        addParticipant("PROPAGATOR", "IMU Propagator", "core");
        addParticipant("TRACKER", "KLT Tracker", "vision");
        addParticipant("UPDATER", "Visual Updater", "core");
        addParticipant("STATE", "State Manager", "core");
    }
    
    void initializeDefaultMessages() {
        addAsyncMessage("IMU_THREAD", "IMU_QUEUE", "push(IMUData)", nullptr);
        addAsyncMessage("VISUAL_THREAD", "IMAGE_QUEUE", "push(ImageData)", nullptr);
        
        addSyncMessage("PROPAGATOR", "IMU_QUEUE", "pop()", "IMUData");
        addSyncMessage("PROPAGATOR", "STATE", "propagate(IMUData)", "void");
        
        addSyncMessage("TRACKER", "IMAGE_QUEUE", "pop()", "ImageData");
        addSyncMessage("TRACKER", "STATE", "getCameraState()", "CameraState");
        addSyncMessage("TRACKER", "TRACKER", "track()", "Features");
        
        addSyncMessage("UPDATER", "STATE", "getFeatures()", "Feature[]");
        addSyncMessage("UPDATER", "UPDATER", "triangulate()", "void");
        addSyncMessage("UPDATER", "STATE", "update(ErrorState)", "void");
    }
    
    void addParticipant(const char* id, const char* name, const char* type) {
        if (participantCount_ < MAX_PARTICIPANTS) {
            participants_[participantCount_].id = id;
            participants_[participantCount_].name = name;
            participants_[participantCount_].type = type;
            participantCount_++;
        }
    }
    
    void addSyncMessage(const char* from, const char* to, 
                       const char* msg, const char* ret) {
        if (messageCount_ < MAX_MESSAGES) {
            messages_[messageCount_].fromId = from;
            messages_[messageCount_].toId = to;
            messages_[messageCount_].message = msg;
            messages_[messageCount_].returnType = ret;
            messages_[messageCount_].isAsync = false;
            messages_[messageCount_].isReturn = false;
            messageCount_++;
        }
    }
    
    void addAsyncMessage(const char* from, const char* to, 
                        const char* msg, const char* ret) {
        if (messageCount_ < MAX_MESSAGES) {
            messages_[messageCount_].fromId = from;
            messages_[messageCount_].toId = to;
            messages_[messageCount_].message = msg;
            messages_[messageCount_].returnType = ret;
            messages_[messageCount_].isAsync = true;
            messages_[messageCount_].isReturn = false;
            messageCount_++;
        }
    }
    
    void generateMermaid(char* buffer, uint32 bufferSize) const {
        uint32 offset = 0;
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "```mermaid\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "sequenceDiagram\n");
        offset += snprintf(buffer + offset, bufferSize - offset,
            "    %% %s\n\n", title_);
        
        for (uint32 i = 0; i < participantCount_; ++i) {
            const SequenceParticipant& p = participants_[i];
            const char* type = "participant";
            if (strcmp(p.type, "thread") == 0) type = "actor";
            else if (strcmp(p.type, "buffer") == 0) type = "participant";
            
            offset += snprintf(buffer + offset, bufferSize - offset,
                "    %s %s as %s\n", type, p.id, p.name);
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset, "\n");
        
        for (uint32 i = 0; i < messageCount_; ++i) {
            const SequenceMessage& m = messages_[i];
            
            if (m.isAsync) {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "    %s -) %s : %s\n",
                    m.fromId, m.toId, m.message);
            } else {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "    %s ->> %s : %s\n",
                    m.fromId, m.toId, m.message);
                
                if (m.returnType && m.returnType[0] != '\0') {
                    offset += snprintf(buffer + offset, bufferSize - offset,
                        "    %s -->> %s : %s\n",
                        m.toId, m.fromId, m.returnType);
                }
            }
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "```\n");
    }
    
    void generatePlantUML(char* buffer, uint32 bufferSize) const {
        uint32 offset = 0;
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "@startuml\n\n");
        
        for (uint32 i = 0; i < participantCount_; ++i) {
            const SequenceParticipant& p = participants_[i];
            offset += snprintf(buffer + offset, bufferSize - offset,
                "participant \"%s\" as %s\n", p.name, p.id);
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset, "\n");
        
        for (uint32 i = 0; i < messageCount_; ++i) {
            const SequenceMessage& m = messages_[i];
            
            if (m.isAsync) {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "%s -> %s : %s\n",
                    m.fromId, m.toId, m.message);
            } else {
                offset += snprintf(buffer + offset, bufferSize - offset,
                    "%s -> %s : %s\n",
                    m.fromId, m.toId, m.message);
                
                if (m.returnType && m.returnType[0] != '\0') {
                    offset += snprintf(buffer + offset, bufferSize - offset,
                        "%s <-- %s : %s\n",
                        m.fromId, m.toId, m.returnType);
                }
            }
        }
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "@enduml\n");
    }
    
    uint32 getParticipantCount() const { return participantCount_; }
    uint32 getMessageCount() const { return messageCount_; }
    const SequenceParticipant& getParticipant(uint32 idx) const { return participants_[idx]; }
    const SequenceMessage& getMessage(uint32 idx) const { return messages_[idx]; }
};

class DocumentationGenerator {
public:
    static void generateAllDiagrams(char* buffer, uint32 bufferSize) {
        uint32 offset = 0;
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "# MSCKF System Documentation\n\n");
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "## 1. Data Flow Diagram\n\n");
        
        DataFlowDiagram dfd;
        char dfdBuffer[4096];
        dfd.generateMermaid(dfdBuffer, sizeof(dfdBuffer));
        offset += snprintf(buffer + offset, bufferSize - offset,
            "%s\n", dfdBuffer);
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "## 2. State Machine Diagram\n\n");
        
        StateMachineDiagram smd;
        char smdBuffer[2048];
        smd.generateMermaid(smdBuffer, sizeof(smdBuffer));
        offset += snprintf(buffer + offset, bufferSize - offset,
            "%s\n", smdBuffer);
        
        offset += snprintf(buffer + offset, bufferSize - offset,
            "## 3. Sequence Diagram\n\n");
        
        SequenceDiagram sd;
        char sdBuffer[2048];
        sd.generateMermaid(sdBuffer, sizeof(sdBuffer));
        offset += snprintf(buffer + offset, bufferSize - offset,
            "%s\n", sdBuffer);
    }
};

}

}

#endif

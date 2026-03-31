# Tracked obstacle with spatial memory state flags
geometry_msgs/Point    center         # Current (or predicted) center position
geometry_msgs/Vector3  velocity       # Last known velocity
float32                radius         # Object radius
int32                  id             # Original tracker ID (-1 if unknown)
bool                   is_ghost       # True: object lost from scan, using predicted pos
bool                   is_unstable    # True: recently stopped/reversed/started (1s armor)
int32                  memory_id      # Internal SpatialMemory ID (stable across tracker resets)
float32                time_since_seen  # Seconds since last confirmed detection

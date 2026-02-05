"""Protobuf event serialization for Kafka.

TIER 6 - Point 26: Protobuf for Kafka Events

This module provides helpers to serialize/deserialize events using Protobuf.
Message size is reduced 40-60% compared to JSON.

To regenerate Python bindings:
    cd backend/app/events/proto
    protoc --python_out=. --pyi_out=. events.proto
"""

import logging
from datetime import datetime
from typing import Any

from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.timestamp_pb2 import Timestamp

# Import generated protobuf classes (after running protoc)
try:
    from app.events.proto import events_pb2
    PROTOBUF_AVAILABLE = True
except ImportError:
    events_pb2 = None
    PROTOBUF_AVAILABLE = False

logger = logging.getLogger(__name__)


def datetime_to_timestamp(dt: datetime) -> Timestamp:
    """Convert Python datetime to Protobuf Timestamp."""
    ts = Timestamp()
    ts.FromDatetime(dt)
    return ts


def timestamp_to_datetime(ts: Timestamp) -> datetime:
    """Convert Protobuf Timestamp to Python datetime."""
    return ts.ToDatetime()


class ProtobufSerializer:
    """Serializer for Protobuf events.
    
    TIER 6 - Point 26: Provides 40-60% smaller message sizes.
    
    Usage:
        serializer = ProtobufSerializer()
        
        # Serialize
        event = {"entity_id": "...", "name": "..."}
        data = serializer.serialize("EntityCreated", event)
        
        # Deserialize
        event = serializer.deserialize("EntityCreated", data)
    """
    
    def __init__(self):
        if not PROTOBUF_AVAILABLE:
            logger.warning(
                "Protobuf classes not available. Run protoc to generate. "
                "Falling back to JSON serialization."
            )
    
    @property
    def available(self) -> bool:
        """Check if Protobuf is available."""
        return PROTOBUF_AVAILABLE
    
    def serialize(self, event_type: str, data: dict[str, Any]) -> bytes:
        """Serialize event to Protobuf bytes.
        
        Args:
            event_type: Event type name (e.g., "EntityCreated")
            data: Event data dict
            
        Returns:
            Serialized bytes
        """
        if not PROTOBUF_AVAILABLE:
            # Fallback to JSON
            import json
            return json.dumps(data).encode()
        
        # Get the message class
        message_class = getattr(events_pb2, event_type, None)
        if message_class is None:
            raise ValueError(f"Unknown event type: {event_type}")
        
        # Convert datetime strings to Timestamp
        data = self._prepare_for_proto(data)
        
        # Create and serialize message
        message = ParseDict(data, message_class())
        return message.SerializeToString()
    
    def deserialize(self, event_type: str, data: bytes) -> dict[str, Any]:
        """Deserialize Protobuf bytes to dict.
        
        Args:
            event_type: Event type name
            data: Serialized bytes
            
        Returns:
            Event data dict
        """
        if not PROTOBUF_AVAILABLE:
            import json
            return json.loads(data.decode())
        
        message_class = getattr(events_pb2, event_type, None)
        if message_class is None:
            raise ValueError(f"Unknown event type: {event_type}")
        
        message = message_class()
        message.ParseFromString(data)
        
        return MessageToDict(message, preserving_proto_field_name=True)
    
    def _prepare_for_proto(self, data: dict) -> dict:
        """Prepare dict for Protobuf conversion."""
        result = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                # Convert to timestamp dict
                result[key] = {
                    "seconds": int(value.timestamp()),
                    "nanos": value.microsecond * 1000,
                }
            elif isinstance(value, dict):
                result[key] = self._prepare_for_proto(value)
            else:
                result[key] = value
        return result
    
    def wrap_event(self, event_type: str, data: dict[str, Any]) -> bytes:
        """Wrap event in ArborEvent envelope.
        
        Use this for polymorphic event handling.
        """
        if not PROTOBUF_AVAILABLE:
            import json
            return json.dumps({"event_type": event_type, "payload": data}).encode()
        
        envelope = events_pb2.ArborEvent()
        envelope.event_type = event_type
        
        # Set the appropriate oneof field
        field_name = self._to_snake_case(event_type)
        inner = getattr(envelope, field_name, None)
        if inner is not None:
            ParseDict(self._prepare_for_proto(data), inner)
        
        return envelope.SerializeToString()
    
    def unwrap_event(self, data: bytes) -> tuple[str, dict[str, Any]]:
        """Unwrap ArborEvent envelope.
        
        Returns:
            Tuple of (event_type, payload)
        """
        if not PROTOBUF_AVAILABLE:
            import json
            d = json.loads(data.decode())
            return d["event_type"], d["payload"]
        
        envelope = events_pb2.ArborEvent()
        envelope.ParseFromString(data)
        
        # Find which oneof field is set
        payload_field = envelope.WhichOneof("payload")
        if payload_field:
            payload = MessageToDict(
                getattr(envelope, payload_field),
                preserving_proto_field_name=True
            )
            return envelope.event_type, payload
        
        return envelope.event_type, {}
    
    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


# Singleton
_serializer: ProtobufSerializer | None = None


def get_proto_serializer() -> ProtobufSerializer:
    """Get singleton ProtobufSerializer."""
    global _serializer
    if _serializer is None:
        _serializer = ProtobufSerializer()
    return _serializer

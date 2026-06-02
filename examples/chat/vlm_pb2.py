# MIT License
#
# Copyright (c) 2026 VinRobotics
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
  name='vlm.proto',
  package='vlm_chat',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\tvlm.proto\x12\x08vlm_chat\"\x82\x01\n\x05Image\x12*\n\x08\x65ncoding\x18\x01 \x01(\x0e\x32\x18.vlm_chat.Image.Encoding\x12\r\n\x05width\x18\x02 \x01(\r\x12\x0e\n\x06height\x18\x03 \x01(\r\x12\x0c\n\x04\x64\x61ta\x18\x04 \x01(\x0c\" \n\x08\x45ncoding\x12\x08\n\x04JPEG\x10\x00\x12\n\n\x06RGB_U8\x10\x01\",\n\x0b\x43hatMessage\x12\x0c\n\x04role\x18\x01 \x01(\t\x12\x0f\n\x07\x63ontent\x18\x02 \x01(\t\"e\n\x0eSamplingParams\x12\x13\n\x0btemperature\x18\x01 \x01(\x02\x12\r\n\x05top_p\x18\x02 \x01(\x02\x12\r\n\x05top_k\x18\x03 \x01(\x05\x12\x12\n\nmax_tokens\x18\x04 \x01(\x05\x12\x0c\n\x04seed\x18\x05 \x01(\x04\"\xa7\x01\n\x0b\x43hatRequest\x12\'\n\x08messages\x18\x01 \x03(\x0b\x32\x15.vlm_chat.ChatMessage\x12\x1f\n\x06images\x18\x02 \x03(\x0b\x32\x0f.vlm_chat.Image\x12*\n\x08sampling\x18\x03 \x01(\x0b\x32\x18.vlm_chat.SamplingParams\x12\x0e\n\x06stream\x18\x04 \x01(\x08\x12\x12\n\nrequest_id\x18\x05 \x01(\x04\"\xd9\x01\n\x0c\x43hatResponse\x12\x12\n\nrequest_id\x18\x01 \x01(\x04\x12\x0c\n\x04text\x18\x02 \x01(\t\x12\x15\n\rfinish_reason\x18\x03 \x01(\t\x12\x15\n\rprompt_tokens\x18\x04 \x01(\r\x12\x19\n\x11\x63ompletion_tokens\x18\x05 \x01(\r\x12\x18\n\x10latency_ms_total\x18\x06 \x01(\x02\x12\x1a\n\x12latency_ms_prefill\x18\x07 \x01(\x02\x12\x19\n\x11latency_ms_decode\x18\x08 \x01(\x02\x12\r\n\x05\x65rror\x18\t \x01(\t\"4\n\x0f\x43hatStreamDelta\x12\x12\n\nrequest_id\x18\x01 \x01(\x04\x12\r\n\x05\x64\x65lta\x18\x02 \x01(\t\"l\n\rStreamMessage\x12*\n\x05\x64\x65lta\x18\x01 \x01(\x0b\x32\x19.vlm_chat.ChatStreamDeltaH\x00\x12\'\n\x05\x66inal\x18\x02 \x01(\x0b\x32\x16.vlm_chat.ChatResponseH\x00\x42\x06\n\x04kindb\x06proto3'
)

_IMAGE_ENCODING = _descriptor.EnumDescriptor(
  name='Encoding',
  full_name='vlm_chat.Image.Encoding',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='JPEG', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='RGB_U8', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=122,
  serialized_end=154,
)
_sym_db.RegisterEnumDescriptor(_IMAGE_ENCODING)

_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='vlm_chat.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='encoding', full_name='vlm_chat.Image.encoding', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='width', full_name='vlm_chat.Image.width', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='height', full_name='vlm_chat.Image.height', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data', full_name='vlm_chat.Image.data', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _IMAGE_ENCODING,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=24,
  serialized_end=154,
)

_CHATMESSAGE = _descriptor.Descriptor(
  name='ChatMessage',
  full_name='vlm_chat.ChatMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='role', full_name='vlm_chat.ChatMessage.role', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='content', full_name='vlm_chat.ChatMessage.content', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=156,
  serialized_end=200,
)

_SAMPLINGPARAMS = _descriptor.Descriptor(
  name='SamplingParams',
  full_name='vlm_chat.SamplingParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='temperature', full_name='vlm_chat.SamplingParams.temperature', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='top_p', full_name='vlm_chat.SamplingParams.top_p', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='top_k', full_name='vlm_chat.SamplingParams.top_k', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='max_tokens', full_name='vlm_chat.SamplingParams.max_tokens', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='seed', full_name='vlm_chat.SamplingParams.seed', index=4,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=202,
  serialized_end=303,
)

_CHATREQUEST = _descriptor.Descriptor(
  name='ChatRequest',
  full_name='vlm_chat.ChatRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='messages', full_name='vlm_chat.ChatRequest.messages', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='images', full_name='vlm_chat.ChatRequest.images', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sampling', full_name='vlm_chat.ChatRequest.sampling', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='stream', full_name='vlm_chat.ChatRequest.stream', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='request_id', full_name='vlm_chat.ChatRequest.request_id', index=4,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=306,
  serialized_end=473,
)

_CHATRESPONSE = _descriptor.Descriptor(
  name='ChatResponse',
  full_name='vlm_chat.ChatResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='request_id', full_name='vlm_chat.ChatResponse.request_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='text', full_name='vlm_chat.ChatResponse.text', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='finish_reason', full_name='vlm_chat.ChatResponse.finish_reason', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='prompt_tokens', full_name='vlm_chat.ChatResponse.prompt_tokens', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='completion_tokens', full_name='vlm_chat.ChatResponse.completion_tokens', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='latency_ms_total', full_name='vlm_chat.ChatResponse.latency_ms_total', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='latency_ms_prefill', full_name='vlm_chat.ChatResponse.latency_ms_prefill', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='latency_ms_decode', full_name='vlm_chat.ChatResponse.latency_ms_decode', index=7,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='error', full_name='vlm_chat.ChatResponse.error', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=476,
  serialized_end=693,
)

_CHATSTREAMDELTA = _descriptor.Descriptor(
  name='ChatStreamDelta',
  full_name='vlm_chat.ChatStreamDelta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='request_id', full_name='vlm_chat.ChatStreamDelta.request_id', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='delta', full_name='vlm_chat.ChatStreamDelta.delta', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=695,
  serialized_end=747,
)

_STREAMMESSAGE = _descriptor.Descriptor(
  name='StreamMessage',
  full_name='vlm_chat.StreamMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='delta', full_name='vlm_chat.StreamMessage.delta', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='final', full_name='vlm_chat.StreamMessage.final', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='kind', full_name='vlm_chat.StreamMessage.kind',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=749,
  serialized_end=857,
)

_IMAGE.fields_by_name['encoding'].enum_type = _IMAGE_ENCODING
_IMAGE_ENCODING.containing_type = _IMAGE
_CHATREQUEST.fields_by_name['messages'].message_type = _CHATMESSAGE
_CHATREQUEST.fields_by_name['images'].message_type = _IMAGE
_CHATREQUEST.fields_by_name['sampling'].message_type = _SAMPLINGPARAMS
_STREAMMESSAGE.fields_by_name['delta'].message_type = _CHATSTREAMDELTA
_STREAMMESSAGE.fields_by_name['final'].message_type = _CHATRESPONSE
_STREAMMESSAGE.oneofs_by_name['kind'].fields.append(
  _STREAMMESSAGE.fields_by_name['delta'])
_STREAMMESSAGE.fields_by_name['delta'].containing_oneof = _STREAMMESSAGE.oneofs_by_name['kind']
_STREAMMESSAGE.oneofs_by_name['kind'].fields.append(
  _STREAMMESSAGE.fields_by_name['final'])
_STREAMMESSAGE.fields_by_name['final'].containing_oneof = _STREAMMESSAGE.oneofs_by_name['kind']
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
DESCRIPTOR.message_types_by_name['ChatMessage'] = _CHATMESSAGE
DESCRIPTOR.message_types_by_name['SamplingParams'] = _SAMPLINGPARAMS
DESCRIPTOR.message_types_by_name['ChatRequest'] = _CHATREQUEST
DESCRIPTOR.message_types_by_name['ChatResponse'] = _CHATRESPONSE
DESCRIPTOR.message_types_by_name['ChatStreamDelta'] = _CHATSTREAMDELTA
DESCRIPTOR.message_types_by_name['StreamMessage'] = _STREAMMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), {
  'DESCRIPTOR' : _IMAGE,
  '__module__' : 'vlm_pb2'

  })
_sym_db.RegisterMessage(Image)

ChatMessage = _reflection.GeneratedProtocolMessageType('ChatMessage', (_message.Message,), {
  'DESCRIPTOR' : _CHATMESSAGE,
  '__module__' : 'vlm_pb2'

  })
_sym_db.RegisterMessage(ChatMessage)

SamplingParams = _reflection.GeneratedProtocolMessageType('SamplingParams', (_message.Message,), {
  'DESCRIPTOR' : _SAMPLINGPARAMS,
  '__module__' : 'vlm_pb2'

  })
_sym_db.RegisterMessage(SamplingParams)

ChatRequest = _reflection.GeneratedProtocolMessageType('ChatRequest', (_message.Message,), {
  'DESCRIPTOR' : _CHATREQUEST,
  '__module__' : 'vlm_pb2'

  })
_sym_db.RegisterMessage(ChatRequest)

ChatResponse = _reflection.GeneratedProtocolMessageType('ChatResponse', (_message.Message,), {
  'DESCRIPTOR' : _CHATRESPONSE,
  '__module__' : 'vlm_pb2'

  })
_sym_db.RegisterMessage(ChatResponse)

ChatStreamDelta = _reflection.GeneratedProtocolMessageType('ChatStreamDelta', (_message.Message,), {
  'DESCRIPTOR' : _CHATSTREAMDELTA,
  '__module__' : 'vlm_pb2'

  })
_sym_db.RegisterMessage(ChatStreamDelta)

StreamMessage = _reflection.GeneratedProtocolMessageType('StreamMessage', (_message.Message,), {
  'DESCRIPTOR' : _STREAMMESSAGE,
  '__module__' : 'vlm_pb2'

  })
_sym_db.RegisterMessage(StreamMessage)

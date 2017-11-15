# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Dataset_ImageSeqGen.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Dataset_ImageSeqGen.proto',
  package='Dataset_ImageSeqGen',
  syntax='proto3',
  serialized_pb=_b('\n\x19\x44\x61taset_ImageSeqGen.proto\x12\x13\x44\x61taset_ImageSeqGen\"c\n\x0c\x44\x61ta_headers\x12\x13\n\x0bimage_width\x18\x01 \x01(\x05\x12\x14\n\x0cimage_height\x18\x02 \x01(\x05\x12\x13\n\x0bimage_depth\x18\x03 \x01(\x05\x12\x13\n\x0bimage_count\x18\x04 \x01(\x05\"X\n\tImage_set\x12\x38\n\rImage_headers\x18\x01 \x01(\x0b\x32!.Dataset_ImageSeqGen.Data_headers\x12\x11\n\tmean_data\x18\x02 \x01(\x0c\x62\x06proto3')
)




_DATA_HEADERS = _descriptor.Descriptor(
  name='Data_headers',
  full_name='Dataset_ImageSeqGen.Data_headers',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='image_width', full_name='Dataset_ImageSeqGen.Data_headers.image_width', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_height', full_name='Dataset_ImageSeqGen.Data_headers.image_height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_depth', full_name='Dataset_ImageSeqGen.Data_headers.image_depth', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_count', full_name='Dataset_ImageSeqGen.Data_headers.image_count', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=149,
)


_IMAGE_SET = _descriptor.Descriptor(
  name='Image_set',
  full_name='Dataset_ImageSeqGen.Image_set',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Image_headers', full_name='Dataset_ImageSeqGen.Image_set.Image_headers', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mean_data', full_name='Dataset_ImageSeqGen.Image_set.mean_data', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=151,
  serialized_end=239,
)

_IMAGE_SET.fields_by_name['Image_headers'].message_type = _DATA_HEADERS
DESCRIPTOR.message_types_by_name['Data_headers'] = _DATA_HEADERS
DESCRIPTOR.message_types_by_name['Image_set'] = _IMAGE_SET
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Data_headers = _reflection.GeneratedProtocolMessageType('Data_headers', (_message.Message,), dict(
  DESCRIPTOR = _DATA_HEADERS,
  __module__ = 'Dataset_ImageSeqGen_pb2'
  # @@protoc_insertion_point(class_scope:Dataset_ImageSeqGen.Data_headers)
  ))
_sym_db.RegisterMessage(Data_headers)

Image_set = _reflection.GeneratedProtocolMessageType('Image_set', (_message.Message,), dict(
  DESCRIPTOR = _IMAGE_SET,
  __module__ = 'Dataset_ImageSeqGen_pb2'
  # @@protoc_insertion_point(class_scope:Dataset_ImageSeqGen.Image_set)
  ))
_sym_db.RegisterMessage(Image_set)


# @@protoc_insertion_point(module_scope)

╩В"
╠п
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceИ
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeКэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
М
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКщшelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.12.02unknown8Щ╨
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
З
gru_1/gru_cell_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╪*&
shared_namegru_1/gru_cell_1/bias
А
)gru_1/gru_cell_1/bias/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/bias*
_output_shapes
:	╪*
dtype0
а
!gru_1/gru_cell_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
╚╪*2
shared_name#!gru_1/gru_cell_1/recurrent_kernel
Щ
5gru_1/gru_cell_1/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_1/gru_cell_1/recurrent_kernel* 
_output_shapes
:
╚╪*
dtype0
М
gru_1/gru_cell_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
╚╪*(
shared_namegru_1/gru_cell_1/kernel
Е
+gru_1/gru_cell_1/kernel/Read/ReadVariableOpReadVariableOpgru_1/gru_cell_1/kernel* 
_output_shapes
:
╚╪*
dtype0

gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╪*"
shared_namegru/gru_cell/bias
x
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes
:	╪*
dtype0
Ш
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
╚╪*.
shared_namegru/gru_cell/recurrent_kernel
С
1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel* 
_output_shapes
:
╚╪*
dtype0
Г
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╪*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:	╪*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	╚*
dtype0
Д
serving_default_gru_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
·
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_inputgru/gru_cell/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru_1/gru_cell_1/biasgru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kerneldense/kernel
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_19998025

NoOpNoOp
Б8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╝7
value▓7Bп7 Bи7
А
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec
#_self_saveable_object_factories*
╩
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _random_generator
#!_self_saveable_object_factories* 
ц
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(_random_generator
)cell
*
state_spec
#+_self_saveable_object_factories*
╩
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator
#3_self_saveable_object_factories* 
╦
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias
#<_self_saveable_object_factories*
<
=0
>1
?2
@3
A4
B5
:6
;7*
<
=0
>1
?2
@3
A4
B5
:6
;7*
* 
░
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Htrace_0
Itrace_1
Jtrace_2
Ktrace_3* 
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
* 
* 

Pserving_default* 
* 

=0
>1
?2*

=0
>1
?2*
* 
Я

Qstates
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Wtrace_0
Xtrace_1
Ytrace_2
Ztrace_3* 
6
[trace_0
\trace_1
]trace_2
^trace_3* 
'
#__self_saveable_object_factories* 
°
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
f_random_generator

=kernel
>recurrent_kernel
?bias
#g_self_saveable_object_factories*
* 
* 
* 
* 
* 
С
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

mtrace_0
ntrace_1* 

otrace_0
ptrace_1* 
'
#q_self_saveable_object_factories* 
* 

@0
A1
B2*

@0
A1
B2*
* 
Я

rstates
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
6
xtrace_0
ytrace_1
ztrace_2
{trace_3* 
6
|trace_0
}trace_1
~trace_2
trace_3* 
(
$А_self_saveable_object_factories* 
А
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses
З_random_generator

@kernel
Arecurrent_kernel
Bbias
$И_self_saveable_object_factories*
* 
* 
* 
* 
* 
Ц
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

Оtrace_0
Пtrace_1* 

Рtrace_0
Сtrace_1* 
(
$Т_self_saveable_object_factories* 
* 

:0
;1*

:0
;1*
* 
Ш
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

Шtrace_0* 

Щtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
SM
VARIABLE_VALUEgru/gru_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEgru/gru_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_1/gru_cell_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!gru_1/gru_cell_1/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_1/gru_cell_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

Ъ0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

=0
>1
?2*

=0
>1
?2*
* 
Ш
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

аtrace_0
бtrace_1* 

вtrace_0
гtrace_1* 
(
$д_self_saveable_object_factories* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

)0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

@0
A1
B2*

@0
A1
B2*
* 
Ю
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses*

кtrace_0
лtrace_1* 

мtrace_0
нtrace_1* 
(
$о_self_saveable_object_factories* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
п	variables
░	keras_api

▒total

▓count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

▒0
▓1*

п	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
щ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biasgru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kernelgru_1/gru_cell_1/biastotalcountConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_save_20000393
ф
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biasgru_1/gru_cell_1/kernel!gru_1/gru_cell_1/recurrent_kernelgru_1/gru_cell_1/biastotalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference__traced_restore_20000433Хў
╜

▌
-__inference_gru_cell_4_layer_call_fn_20000112

inputs
states_0
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
identity

identity_1ИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_19996464p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ╚: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╚
"
_user_specified_name
states_0
ч
╒
H__inference_sequential_layer_call_and_return_conditional_losses_19997865

inputs
gru_19997843:	╪
gru_19997845:	╪ 
gru_19997847:
╚╪!
gru_1_19997851:	╪"
gru_1_19997853:
╚╪"
gru_1_19997855:
╚╪!
dense_19997859:	╚
dense_19997861:
identityИвdense/StatefulPartitionedCallвgru/StatefulPartitionedCallвgru_1/StatefulPartitionedCallў
gru/StatefulPartitionedCallStatefulPartitionedCallinputsgru_19997843gru_19997845gru_19997847*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_19997603█
dropout/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_19997615Ч
gru_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0gru_1_19997851gru_1_19997853gru_1_19997855*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_1_layer_call_and_return_conditional_losses_19997771▌
dropout_1/PartitionedCallPartitionedCall&gru_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_19997783Ж
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_19997859dense_19997861*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_19997441u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         д
NoOpNoOp^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Н"
└
while_body_19996960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_5_19996982_0:	╪/
while_gru_cell_5_19996984_0:
╚╪/
while_gru_cell_5_19996986_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_5_19996982:	╪-
while_gru_cell_5_19996984:
╚╪-
while_gru_cell_5_19996986:
╚╪Ив(while/gru_cell_5/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0Й
(while/gru_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_5_19996982_0while_gru_cell_5_19996984_0while_gru_cell_5_19996986_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_19996946r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : В
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:01while/gru_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: П
while/Identity_4Identity1while/gru_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ╚w

while/NoOpNoOp)^while/gru_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_5_19996982while_gru_cell_5_19996982_0"8
while_gru_cell_5_19996984while_gru_cell_5_19996984_0"8
while_gru_cell_5_19996986while_gru_cell_5_19996986_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2T
(while/gru_cell_5/StatefulPartitionedCall(while/gru_cell_5/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
БM
В
A__inference_gru_layer_call_and_return_conditional_losses_19997603

inputs5
"gru_cell_4_readvariableop_resource:	╪<
)gru_cell_4_matmul_readvariableop_resource:	╪?
+gru_cell_4_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask}
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЛ
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0Т
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19997514*
condR
while_cond_19997513*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ╚▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╝

d
E__inference_dropout_layer_call_and_return_conditional_losses_19999383

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ╚Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧С
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ╚*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?л
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╚T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ш
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:         ╚f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:         ╚"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
╝	
А
gru_while_cond_19998456$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1>
:gru_while_gru_while_cond_19998456___redundant_placeholder0>
:gru_while_gru_while_cond_19998456___redundant_placeholder1>
:gru_while_gru_while_cond_19998456___redundant_placeholder2>
:gru_while_gru_while_cond_19998456___redundant_placeholder3
gru_while_identity
r
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: S
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: "1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::N J

_output_shapes
: 
0
_user_specified_namegru/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namegru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
╩	
ї
C__inference_dense_layer_call_and_return_conditional_losses_20000098

inputs1
matmul_readvariableop_resource:	╚-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
╪╦
┴
#__inference__wrapped_model_19996397
	gru_inputD
1sequential_gru_gru_cell_4_readvariableop_resource:	╪K
8sequential_gru_gru_cell_4_matmul_readvariableop_resource:	╪N
:sequential_gru_gru_cell_4_matmul_1_readvariableop_resource:
╚╪F
3sequential_gru_1_gru_cell_5_readvariableop_resource:	╪N
:sequential_gru_1_gru_cell_5_matmul_readvariableop_resource:
╚╪P
<sequential_gru_1_gru_cell_5_matmul_1_readvariableop_resource:
╚╪B
/sequential_dense_matmul_readvariableop_resource:	╚>
0sequential_dense_biasadd_readvariableop_resource:
identityИв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв/sequential/gru/gru_cell_4/MatMul/ReadVariableOpв1sequential/gru/gru_cell_4/MatMul_1/ReadVariableOpв(sequential/gru/gru_cell_4/ReadVariableOpвsequential/gru/whileв1sequential/gru_1/gru_cell_5/MatMul/ReadVariableOpв3sequential/gru_1/gru_cell_5/MatMul_1/ReadVariableOpв*sequential/gru_1/gru_cell_5/ReadVariableOpвsequential/gru_1/while[
sequential/gru/ShapeShape	gru_input*
T0*
_output_shapes
::э╧l
"sequential/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$sequential/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$sequential/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
sequential/gru/strided_sliceStridedSlicesequential/gru/Shape:output:0+sequential/gru/strided_slice/stack:output:0-sequential/gru/strided_slice/stack_1:output:0-sequential/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
sequential/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚а
sequential/gru/zeros/packedPack%sequential/gru/strided_slice:output:0&sequential/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:_
sequential/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
sequential/gru/zerosFill$sequential/gru/zeros/packed:output:0#sequential/gru/zeros/Const:output:0*
T0*(
_output_shapes
:         ╚r
sequential/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          О
sequential/gru/transpose	Transpose	gru_input&sequential/gru/transpose/perm:output:0*
T0*+
_output_shapes
:         p
sequential/gru/Shape_1Shapesequential/gru/transpose:y:0*
T0*
_output_shapes
::э╧n
$sequential/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&sequential/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&sequential/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
sequential/gru/strided_slice_1StridedSlicesequential/gru/Shape_1:output:0-sequential/gru/strided_slice_1/stack:output:0/sequential/gru/strided_slice_1/stack_1:output:0/sequential/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*sequential/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         с
sequential/gru/TensorArrayV2TensorListReserve3sequential/gru/TensorArrayV2/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Х
Dsequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Н
6sequential/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/gru/transpose:y:0Msequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥n
$sequential/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&sequential/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&sequential/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
sequential/gru/strided_slice_2StridedSlicesequential/gru/transpose:y:0-sequential/gru/strided_slice_2/stack:output:0/sequential/gru/strided_slice_2/stack_1:output:0/sequential/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЫ
(sequential/gru/gru_cell_4/ReadVariableOpReadVariableOp1sequential_gru_gru_cell_4_readvariableop_resource*
_output_shapes
:	╪*
dtype0Х
!sequential/gru/gru_cell_4/unstackUnpack0sequential/gru/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numй
/sequential/gru/gru_cell_4/MatMul/ReadVariableOpReadVariableOp8sequential_gru_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0┐
 sequential/gru/gru_cell_4/MatMulMatMul'sequential/gru/strided_slice_2:output:07sequential/gru/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪╖
!sequential/gru/gru_cell_4/BiasAddBiasAdd*sequential/gru/gru_cell_4/MatMul:product:0*sequential/gru/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪t
)sequential/gru/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ї
sequential/gru/gru_cell_4/splitSplit2sequential/gru/gru_cell_4/split/split_dim:output:0*sequential/gru/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitо
1sequential/gru/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp:sequential_gru_gru_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0╣
"sequential/gru/gru_cell_4/MatMul_1MatMulsequential/gru/zeros:output:09sequential/gru/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪╗
#sequential/gru/gru_cell_4/BiasAdd_1BiasAdd,sequential/gru/gru_cell_4/MatMul_1:product:0*sequential/gru/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪t
sequential/gru/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       v
+sequential/gru/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ▒
!sequential/gru/gru_cell_4/split_1SplitV,sequential/gru/gru_cell_4/BiasAdd_1:output:0(sequential/gru/gru_cell_4/Const:output:04sequential/gru/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitп
sequential/gru/gru_cell_4/addAddV2(sequential/gru/gru_cell_4/split:output:0*sequential/gru/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚В
!sequential/gru/gru_cell_4/SigmoidSigmoid!sequential/gru/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚▒
sequential/gru/gru_cell_4/add_1AddV2(sequential/gru/gru_cell_4/split:output:1*sequential/gru/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚Ж
#sequential/gru/gru_cell_4/Sigmoid_1Sigmoid#sequential/gru/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚м
sequential/gru/gru_cell_4/mulMul'sequential/gru/gru_cell_4/Sigmoid_1:y:0*sequential/gru/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚и
sequential/gru/gru_cell_4/add_2AddV2(sequential/gru/gru_cell_4/split:output:2!sequential/gru/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚~
sequential/gru/gru_cell_4/ReluRelu#sequential/gru/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚Я
sequential/gru/gru_cell_4/mul_1Mul%sequential/gru/gru_cell_4/Sigmoid:y:0sequential/gru/zeros:output:0*
T0*(
_output_shapes
:         ╚d
sequential/gru/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?и
sequential/gru/gru_cell_4/subSub(sequential/gru/gru_cell_4/sub/x:output:0%sequential/gru/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚к
sequential/gru/gru_cell_4/mul_2Mul!sequential/gru/gru_cell_4/sub:z:0,sequential/gru/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚е
sequential/gru/gru_cell_4/add_3AddV2#sequential/gru/gru_cell_4/mul_1:z:0#sequential/gru/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚}
,sequential/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   х
sequential/gru/TensorArrayV2_1TensorListReserve5sequential/gru/TensorArrayV2_1/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥U
sequential/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'sequential/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         c
!sequential/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Д
sequential/gru/whileWhile*sequential/gru/while/loop_counter:output:00sequential/gru/while/maximum_iterations:output:0sequential/gru/time:output:0'sequential/gru/TensorArrayV2_1:handle:0sequential/gru/zeros:output:0'sequential/gru/strided_slice_1:output:0Fsequential/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:01sequential_gru_gru_cell_4_readvariableop_resource8sequential_gru_gru_cell_4_matmul_readvariableop_resource:sequential_gru_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *.
body&R$
"sequential_gru_while_body_19996149*.
cond&R$
"sequential_gru_while_cond_19996148*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Р
?sequential/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   Ё
1sequential/gru/TensorArrayV2Stack/TensorListStackTensorListStacksequential/gru/while:output:3Hsequential/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0w
$sequential/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         p
&sequential/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&sequential/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
sequential/gru/strided_slice_3StridedSlice:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0-sequential/gru/strided_slice_3/stack:output:0/sequential/gru/strided_slice_3/stack_1:output:0/sequential/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maskt
sequential/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ─
sequential/gru/transpose_1	Transpose:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0(sequential/gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚j
sequential/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ~
sequential/dropout/IdentityIdentitysequential/gru/transpose_1:y:0*
T0*,
_output_shapes
:         ╚x
sequential/gru_1/ShapeShape$sequential/dropout/Identity:output:0*
T0*
_output_shapes
::э╧n
$sequential/gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&sequential/gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&sequential/gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
sequential/gru_1/strided_sliceStridedSlicesequential/gru_1/Shape:output:0-sequential/gru_1/strided_slice/stack:output:0/sequential/gru_1/strided_slice/stack_1:output:0/sequential/gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
sequential/gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚ж
sequential/gru_1/zeros/packedPack'sequential/gru_1/strided_slice:output:0(sequential/gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
sequential/gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    а
sequential/gru_1/zerosFill&sequential/gru_1/zeros/packed:output:0%sequential/gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:         ╚t
sequential/gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          о
sequential/gru_1/transpose	Transpose$sequential/dropout/Identity:output:0(sequential/gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:         ╚t
sequential/gru_1/Shape_1Shapesequential/gru_1/transpose:y:0*
T0*
_output_shapes
::э╧p
&sequential/gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential/gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential/gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 sequential/gru_1/strided_slice_1StridedSlice!sequential/gru_1/Shape_1:output:0/sequential/gru_1/strided_slice_1/stack:output:01sequential/gru_1/strided_slice_1/stack_1:output:01sequential/gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,sequential/gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ч
sequential/gru_1/TensorArrayV2TensorListReserve5sequential/gru_1/TensorArrayV2/element_shape:output:0)sequential/gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ч
Fsequential/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   У
8sequential/gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/gru_1/transpose:y:0Osequential/gru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥p
&sequential/gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential/gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential/gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
 sequential/gru_1/strided_slice_2StridedSlicesequential/gru_1/transpose:y:0/sequential/gru_1/strided_slice_2/stack:output:01sequential/gru_1/strided_slice_2/stack_1:output:01sequential/gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maskЯ
*sequential/gru_1/gru_cell_5/ReadVariableOpReadVariableOp3sequential_gru_1_gru_cell_5_readvariableop_resource*
_output_shapes
:	╪*
dtype0Щ
#sequential/gru_1/gru_cell_5/unstackUnpack2sequential/gru_1/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numо
1sequential/gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOp:sequential_gru_1_gru_cell_5_matmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0┼
"sequential/gru_1/gru_cell_5/MatMulMatMul)sequential/gru_1/strided_slice_2:output:09sequential/gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪╜
#sequential/gru_1/gru_cell_5/BiasAddBiasAdd,sequential/gru_1/gru_cell_5/MatMul:product:0,sequential/gru_1/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪v
+sequential/gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ·
!sequential/gru_1/gru_cell_5/splitSplit4sequential/gru_1/gru_cell_5/split/split_dim:output:0,sequential/gru_1/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_split▓
3sequential/gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp<sequential_gru_1_gru_cell_5_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0┐
$sequential/gru_1/gru_cell_5/MatMul_1MatMulsequential/gru_1/zeros:output:0;sequential/gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪┴
%sequential/gru_1/gru_cell_5/BiasAdd_1BiasAdd.sequential/gru_1/gru_cell_5/MatMul_1:product:0,sequential/gru_1/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪v
!sequential/gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       x
-sequential/gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╣
#sequential/gru_1/gru_cell_5/split_1SplitV.sequential/gru_1/gru_cell_5/BiasAdd_1:output:0*sequential/gru_1/gru_cell_5/Const:output:06sequential/gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_split╡
sequential/gru_1/gru_cell_5/addAddV2*sequential/gru_1/gru_cell_5/split:output:0,sequential/gru_1/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚Ж
#sequential/gru_1/gru_cell_5/SigmoidSigmoid#sequential/gru_1/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚╖
!sequential/gru_1/gru_cell_5/add_1AddV2*sequential/gru_1/gru_cell_5/split:output:1,sequential/gru_1/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚К
%sequential/gru_1/gru_cell_5/Sigmoid_1Sigmoid%sequential/gru_1/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚▓
sequential/gru_1/gru_cell_5/mulMul)sequential/gru_1/gru_cell_5/Sigmoid_1:y:0,sequential/gru_1/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚о
!sequential/gru_1/gru_cell_5/add_2AddV2*sequential/gru_1/gru_cell_5/split:output:2#sequential/gru_1/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚В
 sequential/gru_1/gru_cell_5/ReluRelu%sequential/gru_1/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚е
!sequential/gru_1/gru_cell_5/mul_1Mul'sequential/gru_1/gru_cell_5/Sigmoid:y:0sequential/gru_1/zeros:output:0*
T0*(
_output_shapes
:         ╚f
!sequential/gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?о
sequential/gru_1/gru_cell_5/subSub*sequential/gru_1/gru_cell_5/sub/x:output:0'sequential/gru_1/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚░
!sequential/gru_1/gru_cell_5/mul_2Mul#sequential/gru_1/gru_cell_5/sub:z:0.sequential/gru_1/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚л
!sequential/gru_1/gru_cell_5/add_3AddV2%sequential/gru_1/gru_cell_5/mul_1:z:0%sequential/gru_1/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚
.sequential/gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   o
-sequential/gru_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :°
 sequential/gru_1/TensorArrayV2_1TensorListReserve7sequential/gru_1/TensorArrayV2_1/element_shape:output:06sequential/gru_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥W
sequential/gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)sequential/gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         e
#sequential/gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ю
sequential/gru_1/whileWhile,sequential/gru_1/while/loop_counter:output:02sequential/gru_1/while/maximum_iterations:output:0sequential/gru_1/time:output:0)sequential/gru_1/TensorArrayV2_1:handle:0sequential/gru_1/zeros:output:0)sequential/gru_1/strided_slice_1:output:0Hsequential/gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:03sequential_gru_1_gru_cell_5_readvariableop_resource:sequential_gru_1_gru_cell_5_matmul_readvariableop_resource<sequential_gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *0
body(R&
$sequential_gru_1_while_body_19996300*0
cond(R&
$sequential_gru_1_while_cond_19996299*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Т
Asequential/gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   К
3sequential/gru_1/TensorArrayV2Stack/TensorListStackTensorListStacksequential/gru_1/while:output:3Jsequential/gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsy
&sequential/gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         r
(sequential/gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(sequential/gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▌
 sequential/gru_1/strided_slice_3StridedSlice<sequential/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/gru_1/strided_slice_3/stack:output:01sequential/gru_1/strided_slice_3/stack_1:output:01sequential/gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maskv
!sequential/gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ╩
sequential/gru_1/transpose_1	Transpose<sequential/gru_1/TensorArrayV2Stack/TensorListStack:tensor:0*sequential/gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚l
sequential/gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    З
sequential/dropout_1/IdentityIdentity)sequential/gru_1/strided_slice_3:output:0*
T0*(
_output_shapes
:         ╚Ч
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype0л
sequential/dense/MatMulMatMul&sequential/dropout_1/Identity:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0й
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         p
IdentityIdentity!sequential/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ё
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp0^sequential/gru/gru_cell_4/MatMul/ReadVariableOp2^sequential/gru/gru_cell_4/MatMul_1/ReadVariableOp)^sequential/gru/gru_cell_4/ReadVariableOp^sequential/gru/while2^sequential/gru_1/gru_cell_5/MatMul/ReadVariableOp4^sequential/gru_1/gru_cell_5/MatMul_1/ReadVariableOp+^sequential/gru_1/gru_cell_5/ReadVariableOp^sequential/gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2b
/sequential/gru/gru_cell_4/MatMul/ReadVariableOp/sequential/gru/gru_cell_4/MatMul/ReadVariableOp2f
1sequential/gru/gru_cell_4/MatMul_1/ReadVariableOp1sequential/gru/gru_cell_4/MatMul_1/ReadVariableOp2T
(sequential/gru/gru_cell_4/ReadVariableOp(sequential/gru/gru_cell_4/ReadVariableOp2,
sequential/gru/whilesequential/gru/while2f
1sequential/gru_1/gru_cell_5/MatMul/ReadVariableOp1sequential/gru_1/gru_cell_5/MatMul/ReadVariableOp2j
3sequential/gru_1/gru_cell_5/MatMul_1/ReadVariableOp3sequential/gru_1/gru_cell_5/MatMul_1/ReadVariableOp2X
*sequential/gru_1/gru_cell_5/ReadVariableOp*sequential/gru_1/gru_cell_5/ReadVariableOp20
sequential/gru_1/whilesequential/gru_1/while:V R
+
_output_shapes
:         
#
_user_specified_name	gru_input
╚
┤
while_cond_19999806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19999806___redundant_placeholder06
2while_while_cond_19999806___redundant_placeholder16
2while_while_cond_19999806___redundant_placeholder26
2while_while_cond_19999806___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
Щ
╜
(__inference_gru_1_layer_call_fn_19999410
inputs_0
unknown:	╪
	unknown_0:
╚╪
	unknown_1:
╚╪
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_1_layer_call_and_return_conditional_losses_19997025p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ╚: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  ╚
"
_user_specified_name
inputs_0
▒=
Г
while_body_19999272
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	╪D
1while_gru_cell_4_matmul_readvariableop_resource_0:	╪G
3while_gru_cell_4_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	╪B
/while_gru_cell_4_matmul_readvariableop_resource:	╪E
1while_gru_cell_4_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЩ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0╢
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
├
Ц
(__inference_dense_layer_call_fn_20000088

inputs
unknown:	╚
	unknown_0:
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_19997441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
║P
┼
"sequential_gru_while_body_19996149:
6sequential_gru_while_sequential_gru_while_loop_counter@
<sequential_gru_while_sequential_gru_while_maximum_iterations$
 sequential_gru_while_placeholder&
"sequential_gru_while_placeholder_1&
"sequential_gru_while_placeholder_29
5sequential_gru_while_sequential_gru_strided_slice_1_0u
qsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0L
9sequential_gru_while_gru_cell_4_readvariableop_resource_0:	╪S
@sequential_gru_while_gru_cell_4_matmul_readvariableop_resource_0:	╪V
Bsequential_gru_while_gru_cell_4_matmul_1_readvariableop_resource_0:
╚╪!
sequential_gru_while_identity#
sequential_gru_while_identity_1#
sequential_gru_while_identity_2#
sequential_gru_while_identity_3#
sequential_gru_while_identity_47
3sequential_gru_while_sequential_gru_strided_slice_1s
osequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensorJ
7sequential_gru_while_gru_cell_4_readvariableop_resource:	╪Q
>sequential_gru_while_gru_cell_4_matmul_readvariableop_resource:	╪T
@sequential_gru_while_gru_cell_4_matmul_1_readvariableop_resource:
╚╪Ив5sequential/gru/while/gru_cell_4/MatMul/ReadVariableOpв7sequential/gru/while/gru_cell_4/MatMul_1/ReadVariableOpв.sequential/gru/while/gru_cell_4/ReadVariableOpЧ
Fsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ё
8sequential/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0 sequential_gru_while_placeholderOsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0й
.sequential/gru/while/gru_cell_4/ReadVariableOpReadVariableOp9sequential_gru_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0б
'sequential/gru/while/gru_cell_4/unstackUnpack6sequential/gru/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
num╖
5sequential/gru/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp@sequential_gru_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0у
&sequential/gru/while/gru_cell_4/MatMulMatMul?sequential/gru/while/TensorArrayV2Read/TensorListGetItem:item:0=sequential/gru/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪╔
'sequential/gru/while/gru_cell_4/BiasAddBiasAdd0sequential/gru/while/gru_cell_4/MatMul:product:00sequential/gru/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪z
/sequential/gru/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ж
%sequential/gru/while/gru_cell_4/splitSplit8sequential/gru/while/gru_cell_4/split/split_dim:output:00sequential/gru/while/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_split╝
7sequential/gru/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOpBsequential_gru_while_gru_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0╩
(sequential/gru/while/gru_cell_4/MatMul_1MatMul"sequential_gru_while_placeholder_2?sequential/gru/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪═
)sequential/gru/while/gru_cell_4/BiasAdd_1BiasAdd2sequential/gru/while/gru_cell_4/MatMul_1:product:00sequential/gru/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪z
%sequential/gru/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       |
1sequential/gru/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
'sequential/gru/while/gru_cell_4/split_1SplitV2sequential/gru/while/gru_cell_4/BiasAdd_1:output:0.sequential/gru/while/gru_cell_4/Const:output:0:sequential/gru/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_split┴
#sequential/gru/while/gru_cell_4/addAddV2.sequential/gru/while/gru_cell_4/split:output:00sequential/gru/while/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚О
'sequential/gru/while/gru_cell_4/SigmoidSigmoid'sequential/gru/while/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚├
%sequential/gru/while/gru_cell_4/add_1AddV2.sequential/gru/while/gru_cell_4/split:output:10sequential/gru/while/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚Т
)sequential/gru/while/gru_cell_4/Sigmoid_1Sigmoid)sequential/gru/while/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚╛
#sequential/gru/while/gru_cell_4/mulMul-sequential/gru/while/gru_cell_4/Sigmoid_1:y:00sequential/gru/while/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚║
%sequential/gru/while/gru_cell_4/add_2AddV2.sequential/gru/while/gru_cell_4/split:output:2'sequential/gru/while/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚К
$sequential/gru/while/gru_cell_4/ReluRelu)sequential/gru/while/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚░
%sequential/gru/while/gru_cell_4/mul_1Mul+sequential/gru/while/gru_cell_4/Sigmoid:y:0"sequential_gru_while_placeholder_2*
T0*(
_output_shapes
:         ╚j
%sequential/gru/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
#sequential/gru/while/gru_cell_4/subSub.sequential/gru/while/gru_cell_4/sub/x:output:0+sequential/gru/while/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚╝
%sequential/gru/while/gru_cell_4/mul_2Mul'sequential/gru/while/gru_cell_4/sub:z:02sequential/gru/while/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚╖
%sequential/gru/while/gru_cell_4/add_3AddV2)sequential/gru/while/gru_cell_4/mul_1:z:0)sequential/gru/while/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚ 
9sequential/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"sequential_gru_while_placeholder_1 sequential_gru_while_placeholder)sequential/gru/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥\
sequential/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Й
sequential/gru/while/addAddV2 sequential_gru_while_placeholder#sequential/gru/while/add/y:output:0*
T0*
_output_shapes
: ^
sequential/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :г
sequential/gru/while/add_1AddV26sequential_gru_while_sequential_gru_while_loop_counter%sequential/gru/while/add_1/y:output:0*
T0*
_output_shapes
: Ж
sequential/gru/while/IdentityIdentitysequential/gru/while/add_1:z:0^sequential/gru/while/NoOp*
T0*
_output_shapes
: ж
sequential/gru/while/Identity_1Identity<sequential_gru_while_sequential_gru_while_maximum_iterations^sequential/gru/while/NoOp*
T0*
_output_shapes
: Ж
sequential/gru/while/Identity_2Identitysequential/gru/while/add:z:0^sequential/gru/while/NoOp*
T0*
_output_shapes
: │
sequential/gru/while/Identity_3IdentityIsequential/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential/gru/while/NoOp*
T0*
_output_shapes
: е
sequential/gru/while/Identity_4Identity)sequential/gru/while/gru_cell_4/add_3:z:0^sequential/gru/while/NoOp*
T0*(
_output_shapes
:         ╚■
sequential/gru/while/NoOpNoOp6^sequential/gru/while/gru_cell_4/MatMul/ReadVariableOp8^sequential/gru/while/gru_cell_4/MatMul_1/ReadVariableOp/^sequential/gru/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Ж
@sequential_gru_while_gru_cell_4_matmul_1_readvariableop_resourceBsequential_gru_while_gru_cell_4_matmul_1_readvariableop_resource_0"В
>sequential_gru_while_gru_cell_4_matmul_readvariableop_resource@sequential_gru_while_gru_cell_4_matmul_readvariableop_resource_0"t
7sequential_gru_while_gru_cell_4_readvariableop_resource9sequential_gru_while_gru_cell_4_readvariableop_resource_0"G
sequential_gru_while_identity&sequential/gru/while/Identity:output:0"K
sequential_gru_while_identity_1(sequential/gru/while/Identity_1:output:0"K
sequential_gru_while_identity_2(sequential/gru/while/Identity_2:output:0"K
sequential_gru_while_identity_3(sequential/gru/while/Identity_3:output:0"K
sequential_gru_while_identity_4(sequential/gru/while/Identity_4:output:0"l
3sequential_gru_while_sequential_gru_strided_slice_15sequential_gru_while_sequential_gru_strided_slice_1_0"ф
osequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensorqsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2n
5sequential/gru/while/gru_cell_4/MatMul/ReadVariableOp5sequential/gru/while/gru_cell_4/MatMul/ReadVariableOp2r
7sequential/gru/while/gru_cell_4/MatMul_1/ReadVariableOp7sequential/gru/while/gru_cell_4/MatMul_1/ReadVariableOp2`
.sequential/gru/while/gru_cell_4/ReadVariableOp.sequential/gru/while/gru_cell_4/ReadVariableOp:Y U

_output_shapes
: 
;
_user_specified_name#!sequential/gru/while/loop_counter:_[

_output_shapes
: 
A
_user_specified_name)'sequential/gru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
к
▌
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_20000165

inputs
states_0*
readvariableop_resource:	╪1
matmul_readvariableop_resource:	╪4
 matmul_1_readvariableop_resource:
╚╪
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	╪*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         ╪Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         ╪Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         ╚N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         ╚c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         ╚R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         ╚^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         ╚Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         ╚J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:         ╚V
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:         ╚J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         ╚\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:         ╚W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         ╚Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ╚: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╚
"
_user_specified_name
states_0
╤>
Е
while_body_19999962
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_5_readvariableop_resource_0:	╪E
1while_gru_cell_5_matmul_readvariableop_resource_0:
╚╪G
3while_gru_cell_5_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_5_readvariableop_resource:	╪C
/while_gru_cell_5_matmul_readvariableop_resource:
╚╪E
1while_gru_cell_5_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_5/MatMul/ReadVariableOpв(while/gru_cell_5/MatMul_1/ReadVariableOpвwhile/gru_cell_5/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0Л
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЪ
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0╢
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
∙
e
,__inference_dropout_1_layer_call_fn_20000057

inputs
identityИвStatefulPartitionedCall├
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_19997429p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
╚
┤
while_cond_19997680
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19997680___redundant_placeholder06
2while_while_cond_19997680___redundant_placeholder16
2while_while_cond_19997680___redundant_placeholder26
2while_while_cond_19997680___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
з
H
,__inference_dropout_1_layer_call_fn_20000062

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_19997783a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ╚"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
ЗT
═	
!__inference__traced_save_20000393
file_prefix6
#read_disablecopyonread_dense_kernel:	╚1
#read_1_disablecopyonread_dense_bias:?
,read_2_disablecopyonread_gru_gru_cell_kernel:	╪J
6read_3_disablecopyonread_gru_gru_cell_recurrent_kernel:
╚╪=
*read_4_disablecopyonread_gru_gru_cell_bias:	╪D
0read_5_disablecopyonread_gru_1_gru_cell_1_kernel:
╚╪N
:read_6_disablecopyonread_gru_1_gru_cell_1_recurrent_kernel:
╚╪A
.read_7_disablecopyonread_gru_1_gru_cell_1_bias:	╪(
read_8_disablecopyonread_total: (
read_9_disablecopyonread_count: 
savev2_const
identity_21ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 а
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	╚*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	╚b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	╚w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 Я
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_2/DisableCopyOnReadDisableCopyOnRead,read_2_disablecopyonread_gru_gru_cell_kernel"/device:CPU:0*
_output_shapes
 н
Read_2/ReadVariableOpReadVariableOp,read_2_disablecopyonread_gru_gru_cell_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	╪*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	╪d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	╪К
Read_3/DisableCopyOnReadDisableCopyOnRead6read_3_disablecopyonread_gru_gru_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╕
Read_3/ReadVariableOpReadVariableOp6read_3_disablecopyonread_gru_gru_cell_recurrent_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
╚╪*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
╚╪e

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:
╚╪~
Read_4/DisableCopyOnReadDisableCopyOnRead*read_4_disablecopyonread_gru_gru_cell_bias"/device:CPU:0*
_output_shapes
 л
Read_4/ReadVariableOpReadVariableOp*read_4_disablecopyonread_gru_gru_cell_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	╪*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	╪d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	╪Д
Read_5/DisableCopyOnReadDisableCopyOnRead0read_5_disablecopyonread_gru_1_gru_cell_1_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_5/ReadVariableOpReadVariableOp0read_5_disablecopyonread_gru_1_gru_cell_1_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
╚╪*
dtype0p
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
╚╪g
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0* 
_output_shapes
:
╚╪О
Read_6/DisableCopyOnReadDisableCopyOnRead:read_6_disablecopyonread_gru_1_gru_cell_1_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_6/ReadVariableOpReadVariableOp:read_6_disablecopyonread_gru_1_gru_cell_1_recurrent_kernel^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
╚╪*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
╚╪g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
╚╪В
Read_7/DisableCopyOnReadDisableCopyOnRead.read_7_disablecopyonread_gru_1_gru_cell_1_bias"/device:CPU:0*
_output_shapes
 п
Read_7/ReadVariableOpReadVariableOp.read_7_disablecopyonread_gru_1_gru_cell_1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	╪*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	╪f
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	╪r
Read_8/DisableCopyOnReadDisableCopyOnReadread_8_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Ц
Read_8/ReadVariableOpReadVariableOpread_8_disablecopyonread_total^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: r
Read_9/DisableCopyOnReadDisableCopyOnReadread_9_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Ц
Read_9/ReadVariableOpReadVariableOpread_9_disablecopyonread_count^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: ╘
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¤
valueєBЁB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHГ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ╣
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_20Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_21IdentityIdentity_20:output:0^NoOp*
T0*
_output_shapes
: ╒
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_21Identity_21:output:0*+
_input_shapes
: : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
▓5
Й
C__inference_gru_1_layer_call_and_return_conditional_losses_19996881

inputs&
gru_cell_5_19996803:	╪'
gru_cell_5_19996805:
╚╪'
gru_cell_5_19996807:
╚╪
identityИв"gru_cell_5/StatefulPartitionedCallвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:                  ╚R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_mask╬
"gru_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_5_19996803gru_cell_5_19996805gru_cell_5_19996807*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_19996802n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Д
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_5_19996803gru_cell_5_19996805gru_cell_5_19996807*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19996816*
condR
while_cond_19996815*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╫
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ╚s
NoOpNoOp#^gru_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ╚: : : 2H
"gru_cell_5/StatefulPartitionedCall"gru_cell_5/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  ╚
 
_user_specified_nameinputs
№
╤
"sequential_gru_while_cond_19996148:
6sequential_gru_while_sequential_gru_while_loop_counter@
<sequential_gru_while_sequential_gru_while_maximum_iterations$
 sequential_gru_while_placeholder&
"sequential_gru_while_placeholder_1&
"sequential_gru_while_placeholder_2<
8sequential_gru_while_less_sequential_gru_strided_slice_1T
Psequential_gru_while_sequential_gru_while_cond_19996148___redundant_placeholder0T
Psequential_gru_while_sequential_gru_while_cond_19996148___redundant_placeholder1T
Psequential_gru_while_sequential_gru_while_cond_19996148___redundant_placeholder2T
Psequential_gru_while_sequential_gru_while_cond_19996148___redundant_placeholder3!
sequential_gru_while_identity
Ю
sequential/gru/while/LessLess sequential_gru_while_placeholder8sequential_gru_while_less_sequential_gru_strided_slice_1*
T0*
_output_shapes
: i
sequential/gru/while/IdentityIdentitysequential/gru/while/Less:z:0*
T0
*
_output_shapes
: "G
sequential_gru_while_identity&sequential/gru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::Y U

_output_shapes
: 
;
_user_specified_name#!sequential/gru/while/loop_counter:_[

_output_shapes
: 
A
_user_specified_name)'sequential/gru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
▒=
Г
while_body_19998813
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	╪D
1while_gru_cell_4_matmul_readvariableop_resource_0:	╪G
3while_gru_cell_4_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	╪B
/while_gru_cell_4_matmul_readvariableop_resource:	╪E
1while_gru_cell_4_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЩ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0╢
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
БM
В
A__inference_gru_layer_call_and_return_conditional_losses_19999361

inputs5
"gru_cell_4_readvariableop_resource:	╪<
)gru_cell_4_matmul_readvariableop_resource:	╪?
+gru_cell_4_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask}
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЛ
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0Т
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19999272*
condR
while_cond_19999271*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ╚▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╚
┤
while_cond_19999271
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19999271___redundant_placeholder06
2while_while_cond_19999271___redundant_placeholder16
2while_while_cond_19999271___redundant_placeholder26
2while_while_cond_19999271___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
╚
┤
while_cond_19998812
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19998812___redundant_placeholder06
2while_while_cond_19998812___redundant_placeholder16
2while_while_cond_19998812___redundant_placeholder26
2while_while_cond_19998812___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
┐M
Д
A__inference_gru_layer_call_and_return_conditional_losses_19999055
inputs_05
"gru_cell_4_readvariableop_resource:	╪<
)gru_cell_4_matmul_readvariableop_resource:	╪?
+gru_cell_4_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask}
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЛ
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0Т
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19998966*
condR
while_cond_19998965*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ╚*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ╚▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
▒=
Г
while_body_19998966
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	╪D
1while_gru_cell_4_matmul_readvariableop_resource_0:	╪G
3while_gru_cell_4_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	╪B
/while_gru_cell_4_matmul_readvariableop_resource:	╪E
1while_gru_cell_4_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЩ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0╢
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
м
║
&__inference_gru_layer_call_fn_19998727
inputs_0
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_19996683}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
к
▌
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_20000204

inputs
states_0*
readvariableop_resource:	╪1
matmul_readvariableop_resource:	╪4
 matmul_1_readvariableop_resource:
╚╪
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	╪*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         ╪Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         ╪Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         ╚N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         ╚c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         ╚R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         ╚^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         ╚Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         ╚J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:         ╚V
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:         ╚J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         ╚\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:         ╚W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         ╚Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ╚: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╚
"
_user_specified_name
states_0
╚
┤
while_cond_19999118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19999118___redundant_placeholder06
2while_while_cond_19999118___redundant_placeholder16
2while_while_cond_19999118___redundant_placeholder26
2while_while_cond_19999118___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
ь
c
E__inference_dropout_layer_call_and_return_conditional_losses_19997615

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:         ╚`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ╚"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
в
█
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_19996606

inputs

states*
readvariableop_resource:	╪1
matmul_readvariableop_resource:	╪4
 matmul_1_readvariableop_resource:
╚╪
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	╪*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         ╪Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         ╪Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         ╚N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         ╚c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         ╚R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         ╚^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         ╚Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         ╚J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:         ╚T
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:         ╚J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         ╚\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:         ╚W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         ╚Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ╚: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ╚
 
_user_specified_namestates
╤>
Е
while_body_19999497
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_5_readvariableop_resource_0:	╪E
1while_gru_cell_5_matmul_readvariableop_resource_0:
╚╪G
3while_gru_cell_5_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_5_readvariableop_resource:	╪C
/while_gru_cell_5_matmul_readvariableop_resource:
╚╪E
1while_gru_cell_5_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_5/MatMul/ReadVariableOpв(while/gru_cell_5/MatMul_1/ReadVariableOpвwhile/gru_cell_5/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0Л
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЪ
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0╢
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
▓N
З
C__inference_gru_1_layer_call_and_return_conditional_losses_19999742
inputs_05
"gru_cell_5_readvariableop_resource:	╪=
)gru_cell_5_matmul_readvariableop_resource:
╚╪?
+gru_cell_5_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_5/MatMul/ReadVariableOpв"gru_cell_5/MatMul_1/ReadVariableOpвgru_cell_5/ReadVariableOpвwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:                  ╚R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_mask}
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numМ
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0Т
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19999652*
condR
while_cond_19999651*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╫
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ╚▓
NoOpNoOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ╚: : : 2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  ╚
"
_user_specified_name
inputs_0
╝	
А
gru_while_cond_19998130$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1>
:gru_while_gru_while_cond_19998130___redundant_placeholder0>
:gru_while_gru_while_cond_19998130___redundant_placeholder1>
:gru_while_gru_while_cond_19998130___redundant_placeholder2>
:gru_while_gru_while_cond_19998130___redundant_placeholder3
gru_while_identity
r
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: S
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: "1
gru_while_identitygru/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::N J

_output_shapes
: 
0
_user_specified_namegru/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namegru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
ю 
╛
while_body_19996477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_4_19996499_0:	╪.
while_gru_cell_4_19996501_0:	╪/
while_gru_cell_4_19996503_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_4_19996499:	╪,
while_gru_cell_4_19996501:	╪-
while_gru_cell_4_19996503:
╚╪Ив(while/gru_cell_4/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Й
(while/gru_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_4_19996499_0while_gru_cell_4_19996501_0while_gru_cell_4_19996503_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_19996464┌
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: П
while/Identity_4Identity1while/gru_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ╚w

while/NoOpNoOp)^while/gru_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_4_19996499while_gru_cell_4_19996499_0"8
while_gru_cell_4_19996501while_gru_cell_4_19996501_0"8
while_gru_cell_4_19996503while_gru_cell_4_19996503_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2T
(while/gru_cell_4/StatefulPartitionedCall(while/gru_cell_4/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
╚
┤
while_cond_19999651
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19999651___redundant_placeholder06
2while_while_cond_19999651___redundant_placeholder16
2while_while_cond_19999651___redundant_placeholder26
2while_while_cond_19999651___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
ж
▄
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_19996802

inputs

states*
readvariableop_resource:	╪2
matmul_readvariableop_resource:
╚╪4
 matmul_1_readvariableop_resource:
╚╪
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	╪*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         ╪Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         ╪Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         ╚N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         ╚c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         ╚R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         ╚^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         ╚Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         ╚J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:         ╚T
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:         ╚J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         ╚\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:         ╚W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         ╚Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         ╚:         ╚: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ╚
 
_user_specified_namestates
о
▐
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_20000310

inputs
states_0*
readvariableop_resource:	╪2
matmul_readvariableop_resource:
╚╪4
 matmul_1_readvariableop_resource:
╚╪
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	╪*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         ╪Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         ╪Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         ╚N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         ╚c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         ╚R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         ╚^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         ╚Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         ╚J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:         ╚V
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:         ╚J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         ╚\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:         ╚W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         ╚Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         ╚:         ╚: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╚
"
_user_specified_name
states_0
Щ
╜
(__inference_gru_1_layer_call_fn_19999399
inputs_0
unknown:	╪
	unknown_0:
╚╪
	unknown_1:
╚╪
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_1_layer_call_and_return_conditional_losses_19996881p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ╚: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  ╚
"
_user_specified_name
inputs_0
└4
Ж
A__inference_gru_layer_call_and_return_conditional_losses_19996541

inputs&
gru_cell_4_19996465:	╪&
gru_cell_4_19996467:	╪'
gru_cell_4_19996469:
╚╪
identityИв"gru_cell_4/StatefulPartitionedCallвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask╬
"gru_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_4_19996465gru_cell_4_19996467gru_cell_4_19996469*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_19996464n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Д
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_4_19996465gru_cell_4_19996467gru_cell_4_19996469*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19996477*
condR
while_cond_19996476*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ╚*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ╚s
NoOpNoOp#^gru_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2H
"gru_cell_4/StatefulPartitionedCall"gru_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
э	
╒
-__inference_sequential_layer_call_fn_19997838
	gru_input
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
	unknown_2:	╪
	unknown_3:
╚╪
	unknown_4:
╚╪
	unknown_5:	╚
	unknown_6:
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_19997819o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:         
#
_user_specified_name	gru_input
└4
Ж
A__inference_gru_layer_call_and_return_conditional_losses_19996683

inputs&
gru_cell_4_19996607:	╪&
gru_cell_4_19996609:	╪'
gru_cell_4_19996611:
╚╪
identityИв"gru_cell_4/StatefulPartitionedCallвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask╬
"gru_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_4_19996607gru_cell_4_19996609gru_cell_4_19996611*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_19996606n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Д
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_4_19996607gru_cell_4_19996609gru_cell_4_19996611*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19996619*
condR
while_cond_19996618*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ╚*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ╚s
NoOpNoOp#^gru_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2H
"gru_cell_4/StatefulPartitionedCall"gru_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Б
╗
(__inference_gru_1_layer_call_fn_19999432

inputs
unknown:	╪
	unknown_0:
╚╪
	unknown_1:
╚╪
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_1_layer_call_and_return_conditional_losses_19997771p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ╚: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
э	
╒
-__inference_sequential_layer_call_fn_19997884
	gru_input
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
	unknown_2:	╪
	unknown_3:
╚╪
	unknown_4:
╚╪
	unknown_5:	╚
	unknown_6:
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_19997865o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:         
#
_user_specified_name	gru_input
ПN
Е
C__inference_gru_1_layer_call_and_return_conditional_losses_19999897

inputs5
"gru_cell_5_readvariableop_resource:	╪=
)gru_cell_5_matmul_readvariableop_resource:
╚╪?
+gru_cell_5_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_5/MatMul/ReadVariableOpв"gru_cell_5/MatMul_1/ReadVariableOpвgru_cell_5/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         ╚R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_mask}
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numМ
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0Т
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19999807*
condR
while_cond_19999806*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╫
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ╚▓
NoOpNoOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ╚: : : 2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
▓N
З
C__inference_gru_1_layer_call_and_return_conditional_losses_19999587
inputs_05
"gru_cell_5_readvariableop_resource:	╪=
)gru_cell_5_matmul_readvariableop_resource:
╚╪?
+gru_cell_5_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_5/MatMul/ReadVariableOpв"gru_cell_5/MatMul_1/ReadVariableOpвgru_cell_5/ReadVariableOpвwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:                  ╚R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_mask}
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numМ
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0Т
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19999497*
condR
while_cond_19999496*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╫
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ╚▓
NoOpNoOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ╚: : : 2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  ╚
"
_user_specified_name
inputs_0
ф	
╥
-__inference_sequential_layer_call_fn_19998046

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
	unknown_2:	╪
	unknown_3:
╚╪
	unknown_4:
╚╪
	unknown_5:	╚
	unknown_6:
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_19997819o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╚
┤
while_cond_19996618
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19996618___redundant_placeholder06
2while_while_cond_19996618___redundant_placeholder16
2while_while_cond_19996618___redundant_placeholder26
2while_while_cond_19996618___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
жF
╣	
gru_1_while_body_19998608(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0C
0gru_1_while_gru_cell_5_readvariableop_resource_0:	╪K
7gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:
╚╪M
9gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
╚╪
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensorA
.gru_1_while_gru_cell_5_readvariableop_resource:	╪I
5gru_1_while_gru_cell_5_matmul_readvariableop_resource:
╚╪K
7gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
╚╪Ив,gru_1/while/gru_cell_5/MatMul/ReadVariableOpв.gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpв%gru_1/while/gru_cell_5/ReadVariableOpО
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ┼
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0Ч
%gru_1/while/gru_cell_5/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0П
gru_1/while/gru_cell_5/unstackUnpack-gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numж
,gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_5_matmul_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0╚
gru_1/while/gru_cell_5/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪о
gru_1/while/gru_cell_5/BiasAddBiasAdd'gru_1/while/gru_cell_5/MatMul:product:0'gru_1/while/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪q
&gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ы
gru_1/while/gru_cell_5/splitSplit/gru_1/while/gru_cell_5/split/split_dim:output:0'gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitк
.gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0п
gru_1/while/gru_cell_5/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪▓
 gru_1/while/gru_cell_5/BiasAdd_1BiasAdd)gru_1/while/gru_cell_5/MatMul_1:product:0'gru_1/while/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪q
gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       s
(gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         е
gru_1/while/gru_cell_5/split_1SplitV)gru_1/while/gru_cell_5/BiasAdd_1:output:0%gru_1/while/gru_cell_5/Const:output:01gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitж
gru_1/while/gru_cell_5/addAddV2%gru_1/while/gru_cell_5/split:output:0'gru_1/while/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚|
gru_1/while/gru_cell_5/SigmoidSigmoidgru_1/while/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚и
gru_1/while/gru_cell_5/add_1AddV2%gru_1/while/gru_cell_5/split:output:1'gru_1/while/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚А
 gru_1/while/gru_cell_5/Sigmoid_1Sigmoid gru_1/while/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚г
gru_1/while/gru_cell_5/mulMul$gru_1/while/gru_cell_5/Sigmoid_1:y:0'gru_1/while/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚Я
gru_1/while/gru_cell_5/add_2AddV2%gru_1/while/gru_cell_5/split:output:2gru_1/while/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚x
gru_1/while/gru_cell_5/ReluRelu gru_1/while/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚Х
gru_1/while/gru_cell_5/mul_1Mul"gru_1/while/gru_cell_5/Sigmoid:y:0gru_1_while_placeholder_2*
T0*(
_output_shapes
:         ╚a
gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Я
gru_1/while/gru_cell_5/subSub%gru_1/while/gru_cell_5/sub/x:output:0"gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚б
gru_1/while/gru_cell_5/mul_2Mulgru_1/while/gru_cell_5/sub:z:0)gru_1/while/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚Ь
gru_1/while/gru_cell_5/add_3AddV2 gru_1/while/gru_cell_5/mul_1:z:0 gru_1/while/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚x
6gru_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Г
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1?gru_1/while/TensorArrayV2Write/TensorListSetItem/index:output:0 gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥S
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: U
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: В
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: Ш
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: К
gru_1/while/Identity_4Identity gru_1/while/gru_cell_5/add_3:z:0^gru_1/while/NoOp*
T0*(
_output_shapes
:         ╚┌
gru_1/while/NoOpNoOp-^gru_1/while/gru_cell_5/MatMul/ReadVariableOp/^gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"t
7gru_1_while_gru_cell_5_matmul_1_readvariableop_resource9gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"p
5gru_1_while_gru_cell_5_matmul_readvariableop_resource7gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"b
.gru_1_while_gru_cell_5_readvariableop_resource0gru_1_while_gru_cell_5_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"└
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2\
,gru_1/while/gru_cell_5/MatMul/ReadVariableOp,gru_1/while/gru_cell_5/MatMul/ReadVariableOp2`
.gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp.gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2N
%gru_1/while/gru_cell_5/ReadVariableOp%gru_1/while/gru_cell_5/ReadVariableOp:P L

_output_shapes
: 
2
_user_specified_namegru_1/while/loop_counter:VR

_output_shapes
: 
8
_user_specified_name gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
╤
Ы
H__inference_sequential_layer_call_and_return_conditional_losses_19997819

inputs
gru_19997797:	╪
gru_19997799:	╪ 
gru_19997801:
╚╪!
gru_1_19997805:	╪"
gru_1_19997807:
╚╪"
gru_1_19997809:
╚╪!
dense_19997813:	╚
dense_19997815:
identityИвdense/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallвgru/StatefulPartitionedCallвgru_1/StatefulPartitionedCallў
gru/StatefulPartitionedCallStatefulPartitionedCallinputsgru_19997797gru_19997799gru_19997801*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_19997233ы
dropout/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_19997253Я
gru_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0gru_1_19997805gru_1_19997807gru_1_19997809*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_1_layer_call_and_return_conditional_losses_19997409П
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_19997429О
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_19997813dense_19997815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_19997441u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ъ
NoOpNoOp^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╢
ў
$sequential_gru_1_while_cond_19996299>
:sequential_gru_1_while_sequential_gru_1_while_loop_counterD
@sequential_gru_1_while_sequential_gru_1_while_maximum_iterations&
"sequential_gru_1_while_placeholder(
$sequential_gru_1_while_placeholder_1(
$sequential_gru_1_while_placeholder_2@
<sequential_gru_1_while_less_sequential_gru_1_strided_slice_1X
Tsequential_gru_1_while_sequential_gru_1_while_cond_19996299___redundant_placeholder0X
Tsequential_gru_1_while_sequential_gru_1_while_cond_19996299___redundant_placeholder1X
Tsequential_gru_1_while_sequential_gru_1_while_cond_19996299___redundant_placeholder2X
Tsequential_gru_1_while_sequential_gru_1_while_cond_19996299___redundant_placeholder3#
sequential_gru_1_while_identity
ж
sequential/gru_1/while/LessLess"sequential_gru_1_while_placeholder<sequential_gru_1_while_less_sequential_gru_1_strided_slice_1*
T0*
_output_shapes
: m
sequential/gru_1/while/IdentityIdentitysequential/gru_1/while/Less:z:0*
T0
*
_output_shapes
: "K
sequential_gru_1_while_identity(sequential/gru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::[ W

_output_shapes
: 
=
_user_specified_name%#sequential/gru_1/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)sequential/gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
╤>
Е
while_body_19997681
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_5_readvariableop_resource_0:	╪E
1while_gru_cell_5_matmul_readvariableop_resource_0:
╚╪G
3while_gru_cell_5_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_5_readvariableop_resource:	╪C
/while_gru_cell_5_matmul_readvariableop_resource:
╚╪E
1while_gru_cell_5_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_5/MatMul/ReadVariableOpв(while/gru_cell_5/MatMul_1/ReadVariableOpвwhile/gru_cell_5/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0Л
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЪ
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0╢
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
╚
┤
while_cond_19999961
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19999961___redundant_placeholder06
2while_while_cond_19999961___redundant_placeholder16
2while_while_cond_19999961___redundant_placeholder26
2while_while_cond_19999961___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
┌
Ю
H__inference_sequential_layer_call_and_return_conditional_losses_19997448
	gru_input
gru_19997234:	╪
gru_19997236:	╪ 
gru_19997238:
╚╪!
gru_1_19997410:	╪"
gru_1_19997412:
╚╪"
gru_1_19997414:
╚╪!
dense_19997442:	╚
dense_19997444:
identityИвdense/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallвgru/StatefulPartitionedCallвgru_1/StatefulPartitionedCall·
gru/StatefulPartitionedCallStatefulPartitionedCall	gru_inputgru_19997234gru_19997236gru_19997238*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_19997233ы
dropout/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_19997253Я
gru_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0gru_1_19997410gru_1_19997412gru_1_19997414*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_1_layer_call_and_return_conditional_losses_19997409П
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_19997429О
dense/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_19997442dense_19997444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_19997441u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ъ
NoOpNoOp^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:V R
+
_output_shapes
:         
#
_user_specified_name	gru_input
┐M
Д
A__inference_gru_layer_call_and_return_conditional_losses_19998902
inputs_05
"gru_cell_4_readvariableop_resource:	╪<
)gru_cell_4_matmul_readvariableop_resource:	╪?
+gru_cell_4_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask}
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЛ
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0Т
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19998813*
condR
while_cond_19998812*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╠
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ╚*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ╚▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
Н"
└
while_body_19996816
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_5_19996838_0:	╪/
while_gru_cell_5_19996840_0:
╚╪/
while_gru_cell_5_19996842_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_5_19996838:	╪-
while_gru_cell_5_19996840:
╚╪-
while_gru_cell_5_19996842:
╚╪Ив(while/gru_cell_5/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0Й
(while/gru_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_5_19996838_0while_gru_cell_5_19996840_0while_gru_cell_5_19996842_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_19996802r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : В
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:01while/gru_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: П
while/Identity_4Identity1while/gru_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ╚w

while/NoOpNoOp)^while/gru_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_5_19996838while_gru_cell_5_19996838_0"8
while_gru_cell_5_19996840while_gru_cell_5_19996840_0"8
while_gru_cell_5_19996842while_gru_cell_5_19996842_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2T
(while/gru_cell_5/StatefulPartitionedCall(while/gru_cell_5/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
БM
В
A__inference_gru_layer_call_and_return_conditional_losses_19999208

inputs5
"gru_cell_4_readvariableop_resource:	╪<
)gru_cell_4_matmul_readvariableop_resource:	╪?
+gru_cell_4_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask}
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЛ
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0Т
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19999119*
condR
while_cond_19999118*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ╚▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
он
Э
H__inference_sequential_layer_call_and_return_conditional_losses_19998705

inputs9
&gru_gru_cell_4_readvariableop_resource:	╪@
-gru_gru_cell_4_matmul_readvariableop_resource:	╪C
/gru_gru_cell_4_matmul_1_readvariableop_resource:
╚╪;
(gru_1_gru_cell_5_readvariableop_resource:	╪C
/gru_1_gru_cell_5_matmul_readvariableop_resource:
╚╪E
1gru_1_gru_cell_5_matmul_1_readvariableop_resource:
╚╪7
$dense_matmul_readvariableop_resource:	╚3
%dense_biasadd_readvariableop_resource:
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpв$gru/gru_cell_4/MatMul/ReadVariableOpв&gru/gru_cell_4/MatMul_1/ReadVariableOpвgru/gru_cell_4/ReadVariableOpв	gru/whileв&gru_1/gru_cell_5/MatMul/ReadVariableOpв(gru_1/gru_cell_5/MatMul_1/ReadVariableOpвgru_1/gru_cell_5/ReadVariableOpвgru_1/whileM
	gru/ShapeShapeinputs*
T0*
_output_shapes
::э╧a
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    y
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:         ╚g
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
gru/transpose	Transposeinputsgru/transpose/perm:output:0*
T0*+
_output_shapes
:         Z
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
::э╧c
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         └
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥К
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ь
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥c
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЕ
gru/gru_cell_4/ReadVariableOpReadVariableOp&gru_gru_cell_4_readvariableop_resource*
_output_shapes
:	╪*
dtype0
gru/gru_cell_4/unstackUnpack%gru/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numУ
$gru/gru_cell_4/MatMul/ReadVariableOpReadVariableOp-gru_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0Ю
gru/gru_cell_4/MatMulMatMulgru/strided_slice_2:output:0,gru/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ц
gru/gru_cell_4/BiasAddBiasAddgru/gru_cell_4/MatMul:product:0gru/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪i
gru/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╙
gru/gru_cell_4/splitSplit'gru/gru_cell_4/split/split_dim:output:0gru/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitШ
&gru/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp/gru_gru_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0Ш
gru/gru_cell_4/MatMul_1MatMulgru/zeros:output:0.gru/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ъ
gru/gru_cell_4/BiasAdd_1BiasAdd!gru/gru_cell_4/MatMul_1:product:0gru/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪i
gru/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       k
 gru/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Е
gru/gru_cell_4/split_1SplitV!gru/gru_cell_4/BiasAdd_1:output:0gru/gru_cell_4/Const:output:0)gru/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitО
gru/gru_cell_4/addAddV2gru/gru_cell_4/split:output:0gru/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚l
gru/gru_cell_4/SigmoidSigmoidgru/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Р
gru/gru_cell_4/add_1AddV2gru/gru_cell_4/split:output:1gru/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚p
gru/gru_cell_4/Sigmoid_1Sigmoidgru/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚Л
gru/gru_cell_4/mulMulgru/gru_cell_4/Sigmoid_1:y:0gru/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚З
gru/gru_cell_4/add_2AddV2gru/gru_cell_4/split:output:2gru/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚h
gru/gru_cell_4/ReluRelugru/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚~
gru/gru_cell_4/mul_1Mulgru/gru_cell_4/Sigmoid:y:0gru/zeros:output:0*
T0*(
_output_shapes
:         ╚Y
gru/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?З
gru/gru_cell_4/subSubgru/gru_cell_4/sub/x:output:0gru/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚Й
gru/gru_cell_4/mul_2Mulgru/gru_cell_4/sub:z:0!gru/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚Д
gru/gru_cell_4/add_3AddV2gru/gru_cell_4/mul_1:z:0gru/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ─
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥J
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : g
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         X
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0&gru_gru_cell_4_readvariableop_resource-gru_gru_cell_4_matmul_readvariableop_resource/gru_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_while_body_19998457*#
condR
gru_while_cond_19998456*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Е
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╧
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0l
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         e
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          г
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚_
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
dropout/IdentityIdentitygru/transpose_1:y:0*
T0*,
_output_shapes
:         ╚b
gru_1/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
::э╧c
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚Е
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:         ╚i
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
gru_1/transpose	Transposedropout/Identity:output:0gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:         ╚^
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
::э╧e
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥М
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   Є
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥e
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maskЙ
gru_1/gru_cell_5/ReadVariableOpReadVariableOp(gru_1_gru_cell_5_readvariableop_resource*
_output_shapes
:	╪*
dtype0Г
gru_1/gru_cell_5/unstackUnpack'gru_1/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numШ
&gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_5_matmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0д
gru_1/gru_cell_5/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
gru_1/gru_cell_5/BiasAddBiasAdd!gru_1/gru_cell_5/MatMul:product:0!gru_1/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
gru_1/gru_cell_5/splitSplit)gru_1/gru_cell_5/split/split_dim:output:0!gru_1/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЬ
(gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_5_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0Ю
gru_1/gru_cell_5/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
gru_1/gru_cell_5/BiasAdd_1BiasAdd#gru_1/gru_cell_5/MatMul_1:product:0!gru_1/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪k
gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
gru_1/gru_cell_5/split_1SplitV#gru_1/gru_cell_5/BiasAdd_1:output:0gru_1/gru_cell_5/Const:output:0+gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
gru_1/gru_cell_5/addAddV2gru_1/gru_cell_5/split:output:0!gru_1/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚p
gru_1/gru_cell_5/SigmoidSigmoidgru_1/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Ц
gru_1/gru_cell_5/add_1AddV2gru_1/gru_cell_5/split:output:1!gru_1/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚t
gru_1/gru_cell_5/Sigmoid_1Sigmoidgru_1/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚С
gru_1/gru_cell_5/mulMulgru_1/gru_cell_5/Sigmoid_1:y:0!gru_1/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
gru_1/gru_cell_5/add_2AddV2gru_1/gru_cell_5/split:output:2gru_1/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚l
gru_1/gru_cell_5/ReluRelugru_1/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚Д
gru_1/gru_cell_5/mul_1Mulgru_1/gru_cell_5/Sigmoid:y:0gru_1/zeros:output:0*
T0*(
_output_shapes
:         ╚[
gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
gru_1/gru_cell_5/subSubgru_1/gru_cell_5/sub/x:output:0gru_1/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
gru_1/gru_cell_5/mul_2Mulgru_1/gru_cell_5/sub:z:0#gru_1/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
gru_1/gru_cell_5/add_3AddV2gru_1/gru_cell_5/mul_1:z:0gru_1/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚t
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   d
"gru_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :╫
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0+gru_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥L

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         Z
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : П
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_5_readvariableop_resource/gru_1_gru_cell_5_matmul_readvariableop_resource1gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_1_while_body_19998608*%
condR
gru_1_while_cond_19998607*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations З
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   щ
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsn
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         g
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚a
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    q
dropout_1/IdentityIdentitygru_1/strided_slice_3:output:0*
T0*(
_output_shapes
:         ╚Б
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype0К
dense/MatMulMatMuldropout_1/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp%^gru/gru_cell_4/MatMul/ReadVariableOp'^gru/gru_cell_4/MatMul_1/ReadVariableOp^gru/gru_cell_4/ReadVariableOp
^gru/while'^gru_1/gru_cell_5/MatMul/ReadVariableOp)^gru_1/gru_cell_5/MatMul_1/ReadVariableOp ^gru_1/gru_cell_5/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2L
$gru/gru_cell_4/MatMul/ReadVariableOp$gru/gru_cell_4/MatMul/ReadVariableOp2P
&gru/gru_cell_4/MatMul_1/ReadVariableOp&gru/gru_cell_4/MatMul_1/ReadVariableOp2>
gru/gru_cell_4/ReadVariableOpgru/gru_cell_4/ReadVariableOp2
	gru/while	gru/while2P
&gru_1/gru_cell_5/MatMul/ReadVariableOp&gru_1/gru_cell_5/MatMul/ReadVariableOp2T
(gru_1/gru_cell_5/MatMul_1/ReadVariableOp(gru_1/gru_cell_5/MatMul_1/ReadVariableOp2B
gru_1/gru_cell_5/ReadVariableOpgru_1/gru_cell_5/ReadVariableOp2
gru_1/whilegru_1/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
└

▐
-__inference_gru_cell_5_layer_call_fn_20000218

inputs
states_0
unknown:	╪
	unknown_0:
╚╪
	unknown_1:
╚╪
identity

identity_1ИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_19996802p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         ╚:         ╚: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╚
"
_user_specified_name
states_0
Ў	
ж
gru_1_while_cond_19998607(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1B
>gru_1_while_gru_1_while_cond_19998607___redundant_placeholder0B
>gru_1_while_gru_1_while_cond_19998607___redundant_placeholder1B
>gru_1_while_gru_1_while_cond_19998607___redundant_placeholder2B
>gru_1_while_gru_1_while_cond_19998607___redundant_placeholder3
gru_1_while_identity
z
gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: W
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_1_while_identitygru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::P L

_output_shapes
: 
2
_user_specified_namegru_1/while/loop_counter:VR

_output_shapes
: 
8
_user_specified_name gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
В
╕
&__inference_gru_layer_call_fn_19998749

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_19997603t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
│
F
*__inference_dropout_layer_call_fn_19999371

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_19997615e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ╚"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
ў╝
Э
H__inference_sequential_layer_call_and_return_conditional_losses_19998393

inputs9
&gru_gru_cell_4_readvariableop_resource:	╪@
-gru_gru_cell_4_matmul_readvariableop_resource:	╪C
/gru_gru_cell_4_matmul_1_readvariableop_resource:
╚╪;
(gru_1_gru_cell_5_readvariableop_resource:	╪C
/gru_1_gru_cell_5_matmul_readvariableop_resource:
╚╪E
1gru_1_gru_cell_5_matmul_1_readvariableop_resource:
╚╪7
$dense_matmul_readvariableop_resource:	╚3
%dense_biasadd_readvariableop_resource:
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpв$gru/gru_cell_4/MatMul/ReadVariableOpв&gru/gru_cell_4/MatMul_1/ReadVariableOpвgru/gru_cell_4/ReadVariableOpв	gru/whileв&gru_1/gru_cell_5/MatMul/ReadVariableOpв(gru_1/gru_cell_5/MatMul_1/ReadVariableOpвgru_1/gru_cell_5/ReadVariableOpвgru_1/whileM
	gru/ShapeShapeinputs*
T0*
_output_shapes
::э╧a
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    y
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:         ╚g
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
gru/transpose	Transposeinputsgru/transpose/perm:output:0*
T0*+
_output_shapes
:         Z
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
::э╧c
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         └
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥К
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ь
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥c
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЕ
gru/gru_cell_4/ReadVariableOpReadVariableOp&gru_gru_cell_4_readvariableop_resource*
_output_shapes
:	╪*
dtype0
gru/gru_cell_4/unstackUnpack%gru/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numУ
$gru/gru_cell_4/MatMul/ReadVariableOpReadVariableOp-gru_gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0Ю
gru/gru_cell_4/MatMulMatMulgru/strided_slice_2:output:0,gru/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ц
gru/gru_cell_4/BiasAddBiasAddgru/gru_cell_4/MatMul:product:0gru/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪i
gru/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╙
gru/gru_cell_4/splitSplit'gru/gru_cell_4/split/split_dim:output:0gru/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitШ
&gru/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp/gru_gru_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0Ш
gru/gru_cell_4/MatMul_1MatMulgru/zeros:output:0.gru/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ъ
gru/gru_cell_4/BiasAdd_1BiasAdd!gru/gru_cell_4/MatMul_1:product:0gru/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪i
gru/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       k
 gru/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Е
gru/gru_cell_4/split_1SplitV!gru/gru_cell_4/BiasAdd_1:output:0gru/gru_cell_4/Const:output:0)gru/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitО
gru/gru_cell_4/addAddV2gru/gru_cell_4/split:output:0gru/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚l
gru/gru_cell_4/SigmoidSigmoidgru/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Р
gru/gru_cell_4/add_1AddV2gru/gru_cell_4/split:output:1gru/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚p
gru/gru_cell_4/Sigmoid_1Sigmoidgru/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚Л
gru/gru_cell_4/mulMulgru/gru_cell_4/Sigmoid_1:y:0gru/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚З
gru/gru_cell_4/add_2AddV2gru/gru_cell_4/split:output:2gru/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚h
gru/gru_cell_4/ReluRelugru/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚~
gru/gru_cell_4/mul_1Mulgru/gru_cell_4/Sigmoid:y:0gru/zeros:output:0*
T0*(
_output_shapes
:         ╚Y
gru/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?З
gru/gru_cell_4/subSubgru/gru_cell_4/sub/x:output:0gru/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚Й
gru/gru_cell_4/mul_2Mulgru/gru_cell_4/sub:z:0!gru/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚Д
gru/gru_cell_4/add_3AddV2gru/gru_cell_4/mul_1:z:0gru/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚r
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ─
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥J
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : g
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         X
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ї
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0&gru_gru_cell_4_readvariableop_resource-gru_gru_cell_4_matmul_readvariableop_resource/gru_gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_while_body_19998131*#
condR
gru_while_cond_19998130*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Е
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╧
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0l
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         e
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maski
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          г
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚_
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ж
dropout/dropout/MulMulgru/transpose_1:y:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:         ╚f
dropout/dropout/ShapeShapegru/transpose_1:y:0*
T0*
_output_shapes
::э╧б
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╚*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?├
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╚\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╕
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*,
_output_shapes
:         ╚j
gru_1/ShapeShape!dropout/dropout/SelectV2:output:0*
T0*
_output_shapes
::э╧c
gru_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
gru_1/strided_sliceStridedSlicegru_1/Shape:output:0"gru_1/strided_slice/stack:output:0$gru_1/strided_slice/stack_1:output:0$gru_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
gru_1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚Е
gru_1/zeros/packedPackgru_1/strided_slice:output:0gru_1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
gru_1/zerosFillgru_1/zeros/packed:output:0gru_1/zeros/Const:output:0*
T0*(
_output_shapes
:         ╚i
gru_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Х
gru_1/transpose	Transpose!dropout/dropout/SelectV2:output:0gru_1/transpose/perm:output:0*
T0*,
_output_shapes
:         ╚^
gru_1/Shape_1Shapegru_1/transpose:y:0*
T0*
_output_shapes
::э╧e
gru_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
gru_1/strided_slice_1StridedSlicegru_1/Shape_1:output:0$gru_1/strided_slice_1/stack:output:0&gru_1/strided_slice_1/stack_1:output:0&gru_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╞
gru_1/TensorArrayV2TensorListReserve*gru_1/TensorArrayV2/element_shape:output:0gru_1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥М
;gru_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   Є
-gru_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_1/transpose:y:0Dgru_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥e
gru_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
gru_1/strided_slice_2StridedSlicegru_1/transpose:y:0$gru_1/strided_slice_2/stack:output:0&gru_1/strided_slice_2/stack_1:output:0&gru_1/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maskЙ
gru_1/gru_cell_5/ReadVariableOpReadVariableOp(gru_1_gru_cell_5_readvariableop_resource*
_output_shapes
:	╪*
dtype0Г
gru_1/gru_cell_5/unstackUnpack'gru_1/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numШ
&gru_1/gru_cell_5/MatMul/ReadVariableOpReadVariableOp/gru_1_gru_cell_5_matmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0д
gru_1/gru_cell_5/MatMulMatMulgru_1/strided_slice_2:output:0.gru_1/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
gru_1/gru_cell_5/BiasAddBiasAdd!gru_1/gru_cell_5/MatMul:product:0!gru_1/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 gru_1/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
gru_1/gru_cell_5/splitSplit)gru_1/gru_cell_5/split/split_dim:output:0!gru_1/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЬ
(gru_1/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp1gru_1_gru_cell_5_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0Ю
gru_1/gru_cell_5/MatMul_1MatMulgru_1/zeros:output:00gru_1/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
gru_1/gru_cell_5/BiasAdd_1BiasAdd#gru_1/gru_cell_5/MatMul_1:product:0!gru_1/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪k
gru_1/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"gru_1/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
gru_1/gru_cell_5/split_1SplitV#gru_1/gru_cell_5/BiasAdd_1:output:0gru_1/gru_cell_5/Const:output:0+gru_1/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
gru_1/gru_cell_5/addAddV2gru_1/gru_cell_5/split:output:0!gru_1/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚p
gru_1/gru_cell_5/SigmoidSigmoidgru_1/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Ц
gru_1/gru_cell_5/add_1AddV2gru_1/gru_cell_5/split:output:1!gru_1/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚t
gru_1/gru_cell_5/Sigmoid_1Sigmoidgru_1/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚С
gru_1/gru_cell_5/mulMulgru_1/gru_cell_5/Sigmoid_1:y:0!gru_1/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
gru_1/gru_cell_5/add_2AddV2gru_1/gru_cell_5/split:output:2gru_1/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚l
gru_1/gru_cell_5/ReluRelugru_1/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚Д
gru_1/gru_cell_5/mul_1Mulgru_1/gru_cell_5/Sigmoid:y:0gru_1/zeros:output:0*
T0*(
_output_shapes
:         ╚[
gru_1/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
gru_1/gru_cell_5/subSubgru_1/gru_cell_5/sub/x:output:0gru_1/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
gru_1/gru_cell_5/mul_2Mulgru_1/gru_cell_5/sub:z:0#gru_1/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
gru_1/gru_cell_5/add_3AddV2gru_1/gru_cell_5/mul_1:z:0gru_1/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚t
#gru_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   d
"gru_1/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :╫
gru_1/TensorArrayV2_1TensorListReserve,gru_1/TensorArrayV2_1/element_shape:output:0+gru_1/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥L

gru_1/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         Z
gru_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : П
gru_1/whileWhile!gru_1/while/loop_counter:output:0'gru_1/while/maximum_iterations:output:0gru_1/time:output:0gru_1/TensorArrayV2_1:handle:0gru_1/zeros:output:0gru_1/strided_slice_1:output:0=gru_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(gru_1_gru_cell_5_readvariableop_resource/gru_1_gru_cell_5_matmul_readvariableop_resource1gru_1_gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *%
bodyR
gru_1_while_body_19998289*%
condR
gru_1_while_cond_19998288*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations З
6gru_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   щ
(gru_1/TensorArrayV2Stack/TensorListStackTensorListStackgru_1/while:output:3?gru_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsn
gru_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         g
gru_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
gru_1/strided_slice_3StridedSlice1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0$gru_1/strided_slice_3/stack:output:0&gru_1/strided_slice_3/stack_1:output:0&gru_1/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maskk
gru_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
gru_1/transpose_1	Transpose1gru_1/TensorArrayV2Stack/TensorListStack:tensor:0gru_1/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚a
gru_1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @С
dropout_1/dropout/MulMulgru_1/strided_slice_3:output:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:         ╚s
dropout_1/dropout/ShapeShapegru_1/strided_slice_3:output:0*
T0*
_output_shapes
::э╧б
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?┼
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚^
dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ╝
dropout_1/dropout/SelectV2SelectV2"dropout_1/dropout/GreaterEqual:z:0dropout_1/dropout/Mul:z:0"dropout_1/dropout/Const_1:output:0*
T0*(
_output_shapes
:         ╚Б
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype0Т
dense/MatMulMatMul#dropout_1/dropout/SelectV2:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Г
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp%^gru/gru_cell_4/MatMul/ReadVariableOp'^gru/gru_cell_4/MatMul_1/ReadVariableOp^gru/gru_cell_4/ReadVariableOp
^gru/while'^gru_1/gru_cell_5/MatMul/ReadVariableOp)^gru_1/gru_cell_5/MatMul_1/ReadVariableOp ^gru_1/gru_cell_5/ReadVariableOp^gru_1/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2L
$gru/gru_cell_4/MatMul/ReadVariableOp$gru/gru_cell_4/MatMul/ReadVariableOp2P
&gru/gru_cell_4/MatMul_1/ReadVariableOp&gru/gru_cell_4/MatMul_1/ReadVariableOp2>
gru/gru_cell_4/ReadVariableOpgru/gru_cell_4/ReadVariableOp2
	gru/while	gru/while2P
&gru_1/gru_cell_5/MatMul/ReadVariableOp&gru_1/gru_cell_5/MatMul/ReadVariableOp2T
(gru_1/gru_cell_5/MatMul_1/ReadVariableOp(gru_1/gru_cell_5/MatMul_1/ReadVariableOp2B
gru_1/gru_cell_5/ReadVariableOpgru_1/gru_cell_5/ReadVariableOp2
gru_1/whilegru_1/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ПN
Е
C__inference_gru_1_layer_call_and_return_conditional_losses_19997771

inputs5
"gru_cell_5_readvariableop_resource:	╪=
)gru_cell_5_matmul_readvariableop_resource:
╚╪?
+gru_cell_5_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_5/MatMul/ReadVariableOpв"gru_cell_5/MatMul_1/ReadVariableOpвgru_cell_5/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         ╚R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_mask}
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numМ
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0Т
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19997681*
condR
while_cond_19997680*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╫
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ╚▓
NoOpNoOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ╚: : : 2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
╚
┤
while_cond_19998965
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19998965___redundant_placeholder06
2while_while_cond_19998965___redundant_placeholder16
2while_while_cond_19998965___redundant_placeholder26
2while_while_cond_19998965___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
Е
c
*__inference_dropout_layer_call_fn_19999366

inputs
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_19997253t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
░T
Г
$sequential_gru_1_while_body_19996300>
:sequential_gru_1_while_sequential_gru_1_while_loop_counterD
@sequential_gru_1_while_sequential_gru_1_while_maximum_iterations&
"sequential_gru_1_while_placeholder(
$sequential_gru_1_while_placeholder_1(
$sequential_gru_1_while_placeholder_2=
9sequential_gru_1_while_sequential_gru_1_strided_slice_1_0y
usequential_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_1_tensorarrayunstack_tensorlistfromtensor_0N
;sequential_gru_1_while_gru_cell_5_readvariableop_resource_0:	╪V
Bsequential_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:
╚╪X
Dsequential_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
╚╪#
sequential_gru_1_while_identity%
!sequential_gru_1_while_identity_1%
!sequential_gru_1_while_identity_2%
!sequential_gru_1_while_identity_3%
!sequential_gru_1_while_identity_4;
7sequential_gru_1_while_sequential_gru_1_strided_slice_1w
ssequential_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_1_tensorarrayunstack_tensorlistfromtensorL
9sequential_gru_1_while_gru_cell_5_readvariableop_resource:	╪T
@sequential_gru_1_while_gru_cell_5_matmul_readvariableop_resource:
╚╪V
Bsequential_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
╚╪Ив7sequential/gru_1/while/gru_cell_5/MatMul/ReadVariableOpв9sequential/gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpв0sequential/gru_1/while/gru_cell_5/ReadVariableOpЩ
Hsequential/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   №
:sequential/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemusequential_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_1_tensorarrayunstack_tensorlistfromtensor_0"sequential_gru_1_while_placeholderQsequential/gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0н
0sequential/gru_1/while/gru_cell_5/ReadVariableOpReadVariableOp;sequential_gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0е
)sequential/gru_1/while/gru_cell_5/unstackUnpack8sequential/gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
num╝
7sequential/gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOpBsequential_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0щ
(sequential/gru_1/while/gru_cell_5/MatMulMatMulAsequential/gru_1/while/TensorArrayV2Read/TensorListGetItem:item:0?sequential/gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪╧
)sequential/gru_1/while/gru_cell_5/BiasAddBiasAdd2sequential/gru_1/while/gru_cell_5/MatMul:product:02sequential/gru_1/while/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪|
1sequential/gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         М
'sequential/gru_1/while/gru_cell_5/splitSplit:sequential/gru_1/while/gru_cell_5/split/split_dim:output:02sequential/gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_split└
9sequential/gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOpDsequential_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0╨
*sequential/gru_1/while/gru_cell_5/MatMul_1MatMul$sequential_gru_1_while_placeholder_2Asequential/gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪╙
+sequential/gru_1/while/gru_cell_5/BiasAdd_1BiasAdd4sequential/gru_1/while/gru_cell_5/MatMul_1:product:02sequential/gru_1/while/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪|
'sequential/gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       ~
3sequential/gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╤
)sequential/gru_1/while/gru_cell_5/split_1SplitV4sequential/gru_1/while/gru_cell_5/BiasAdd_1:output:00sequential/gru_1/while/gru_cell_5/Const:output:0<sequential/gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_split╟
%sequential/gru_1/while/gru_cell_5/addAddV20sequential/gru_1/while/gru_cell_5/split:output:02sequential/gru_1/while/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚Т
)sequential/gru_1/while/gru_cell_5/SigmoidSigmoid)sequential/gru_1/while/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚╔
'sequential/gru_1/while/gru_cell_5/add_1AddV20sequential/gru_1/while/gru_cell_5/split:output:12sequential/gru_1/while/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚Ц
+sequential/gru_1/while/gru_cell_5/Sigmoid_1Sigmoid+sequential/gru_1/while/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚─
%sequential/gru_1/while/gru_cell_5/mulMul/sequential/gru_1/while/gru_cell_5/Sigmoid_1:y:02sequential/gru_1/while/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚└
'sequential/gru_1/while/gru_cell_5/add_2AddV20sequential/gru_1/while/gru_cell_5/split:output:2)sequential/gru_1/while/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚О
&sequential/gru_1/while/gru_cell_5/ReluRelu+sequential/gru_1/while/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚╢
'sequential/gru_1/while/gru_cell_5/mul_1Mul-sequential/gru_1/while/gru_cell_5/Sigmoid:y:0$sequential_gru_1_while_placeholder_2*
T0*(
_output_shapes
:         ╚l
'sequential/gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?└
%sequential/gru_1/while/gru_cell_5/subSub0sequential/gru_1/while/gru_cell_5/sub/x:output:0-sequential/gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚┬
'sequential/gru_1/while/gru_cell_5/mul_2Mul)sequential/gru_1/while/gru_cell_5/sub:z:04sequential/gru_1/while/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚╜
'sequential/gru_1/while/gru_cell_5/add_3AddV2+sequential/gru_1/while/gru_cell_5/mul_1:z:0+sequential/gru_1/while/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚Г
Asequential/gru_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : п
;sequential/gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$sequential_gru_1_while_placeholder_1Jsequential/gru_1/while/TensorArrayV2Write/TensorListSetItem/index:output:0+sequential/gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥^
sequential/gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :П
sequential/gru_1/while/addAddV2"sequential_gru_1_while_placeholder%sequential/gru_1/while/add/y:output:0*
T0*
_output_shapes
: `
sequential/gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :л
sequential/gru_1/while/add_1AddV2:sequential_gru_1_while_sequential_gru_1_while_loop_counter'sequential/gru_1/while/add_1/y:output:0*
T0*
_output_shapes
: М
sequential/gru_1/while/IdentityIdentity sequential/gru_1/while/add_1:z:0^sequential/gru_1/while/NoOp*
T0*
_output_shapes
: о
!sequential/gru_1/while/Identity_1Identity@sequential_gru_1_while_sequential_gru_1_while_maximum_iterations^sequential/gru_1/while/NoOp*
T0*
_output_shapes
: М
!sequential/gru_1/while/Identity_2Identitysequential/gru_1/while/add:z:0^sequential/gru_1/while/NoOp*
T0*
_output_shapes
: ╣
!sequential/gru_1/while/Identity_3IdentityKsequential/gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential/gru_1/while/NoOp*
T0*
_output_shapes
: л
!sequential/gru_1/while/Identity_4Identity+sequential/gru_1/while/gru_cell_5/add_3:z:0^sequential/gru_1/while/NoOp*
T0*(
_output_shapes
:         ╚Ж
sequential/gru_1/while/NoOpNoOp8^sequential/gru_1/while/gru_cell_5/MatMul/ReadVariableOp:^sequential/gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp1^sequential/gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "К
Bsequential_gru_1_while_gru_cell_5_matmul_1_readvariableop_resourceDsequential_gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"Ж
@sequential_gru_1_while_gru_cell_5_matmul_readvariableop_resourceBsequential_gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"x
9sequential_gru_1_while_gru_cell_5_readvariableop_resource;sequential_gru_1_while_gru_cell_5_readvariableop_resource_0"K
sequential_gru_1_while_identity(sequential/gru_1/while/Identity:output:0"O
!sequential_gru_1_while_identity_1*sequential/gru_1/while/Identity_1:output:0"O
!sequential_gru_1_while_identity_2*sequential/gru_1/while/Identity_2:output:0"O
!sequential_gru_1_while_identity_3*sequential/gru_1/while/Identity_3:output:0"O
!sequential_gru_1_while_identity_4*sequential/gru_1/while/Identity_4:output:0"t
7sequential_gru_1_while_sequential_gru_1_strided_slice_19sequential_gru_1_while_sequential_gru_1_strided_slice_1_0"ь
ssequential_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_1_tensorarrayunstack_tensorlistfromtensorusequential_gru_1_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2r
7sequential/gru_1/while/gru_cell_5/MatMul/ReadVariableOp7sequential/gru_1/while/gru_cell_5/MatMul/ReadVariableOp2v
9sequential/gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp9sequential/gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2d
0sequential/gru_1/while/gru_cell_5/ReadVariableOp0sequential/gru_1/while/gru_cell_5/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#sequential/gru_1/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)sequential/gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
В
╕
&__inference_gru_layer_call_fn_19998738

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_19997233t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▒=
Г
while_body_19997144
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	╪D
1while_gru_cell_4_matmul_readvariableop_resource_0:	╪G
3while_gru_cell_4_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	╪B
/while_gru_cell_4_matmul_readvariableop_resource:	╪E
1while_gru_cell_4_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЩ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0╢
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
╚
┤
while_cond_19996476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19996476___redundant_placeholder06
2while_while_cond_19996476___redundant_placeholder16
2while_while_cond_19996476___redundant_placeholder26
2while_while_cond_19996476___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
▒=
Г
while_body_19997514
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	╪D
1while_gru_cell_4_matmul_readvariableop_resource_0:	╪G
3while_gru_cell_4_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	╪B
/while_gru_cell_4_matmul_readvariableop_resource:	╪E
1while_gru_cell_4_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЩ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0╢
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
о
▐
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_20000271

inputs
states_0*
readvariableop_resource:	╪2
matmul_readvariableop_resource:
╚╪4
 matmul_1_readvariableop_resource:
╚╪
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	╪*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         ╪Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         ╪Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         ╚N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         ╚c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         ╚R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         ╚^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         ╚Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         ╚J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:         ╚V
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:         ╚J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         ╚\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:         ╚W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         ╚Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         ╚:         ╚: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╚
"
_user_specified_name
states_0
жF
╣	
gru_1_while_body_19998289(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2'
#gru_1_while_gru_1_strided_slice_1_0c
_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0C
0gru_1_while_gru_cell_5_readvariableop_resource_0:	╪K
7gru_1_while_gru_cell_5_matmul_readvariableop_resource_0:
╚╪M
9gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0:
╚╪
gru_1_while_identity
gru_1_while_identity_1
gru_1_while_identity_2
gru_1_while_identity_3
gru_1_while_identity_4%
!gru_1_while_gru_1_strided_slice_1a
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensorA
.gru_1_while_gru_cell_5_readvariableop_resource:	╪I
5gru_1_while_gru_cell_5_matmul_readvariableop_resource:
╚╪K
7gru_1_while_gru_cell_5_matmul_1_readvariableop_resource:
╚╪Ив,gru_1/while/gru_cell_5/MatMul/ReadVariableOpв.gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpв%gru_1/while/gru_cell_5/ReadVariableOpО
=gru_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ┼
/gru_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0gru_1_while_placeholderFgru_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0Ч
%gru_1/while/gru_cell_5/ReadVariableOpReadVariableOp0gru_1_while_gru_cell_5_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0П
gru_1/while/gru_cell_5/unstackUnpack-gru_1/while/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numж
,gru_1/while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp7gru_1_while_gru_cell_5_matmul_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0╚
gru_1/while/gru_cell_5/MatMulMatMul6gru_1/while/TensorArrayV2Read/TensorListGetItem:item:04gru_1/while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪о
gru_1/while/gru_cell_5/BiasAddBiasAdd'gru_1/while/gru_cell_5/MatMul:product:0'gru_1/while/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪q
&gru_1/while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ы
gru_1/while/gru_cell_5/splitSplit/gru_1/while/gru_cell_5/split/split_dim:output:0'gru_1/while/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitк
.gru_1/while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp9gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0п
gru_1/while/gru_cell_5/MatMul_1MatMulgru_1_while_placeholder_26gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪▓
 gru_1/while/gru_cell_5/BiasAdd_1BiasAdd)gru_1/while/gru_cell_5/MatMul_1:product:0'gru_1/while/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪q
gru_1/while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       s
(gru_1/while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         е
gru_1/while/gru_cell_5/split_1SplitV)gru_1/while/gru_cell_5/BiasAdd_1:output:0%gru_1/while/gru_cell_5/Const:output:01gru_1/while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitж
gru_1/while/gru_cell_5/addAddV2%gru_1/while/gru_cell_5/split:output:0'gru_1/while/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚|
gru_1/while/gru_cell_5/SigmoidSigmoidgru_1/while/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚и
gru_1/while/gru_cell_5/add_1AddV2%gru_1/while/gru_cell_5/split:output:1'gru_1/while/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚А
 gru_1/while/gru_cell_5/Sigmoid_1Sigmoid gru_1/while/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚г
gru_1/while/gru_cell_5/mulMul$gru_1/while/gru_cell_5/Sigmoid_1:y:0'gru_1/while/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚Я
gru_1/while/gru_cell_5/add_2AddV2%gru_1/while/gru_cell_5/split:output:2gru_1/while/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚x
gru_1/while/gru_cell_5/ReluRelu gru_1/while/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚Х
gru_1/while/gru_cell_5/mul_1Mul"gru_1/while/gru_cell_5/Sigmoid:y:0gru_1_while_placeholder_2*
T0*(
_output_shapes
:         ╚a
gru_1/while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Я
gru_1/while/gru_cell_5/subSub%gru_1/while/gru_cell_5/sub/x:output:0"gru_1/while/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚б
gru_1/while/gru_cell_5/mul_2Mulgru_1/while/gru_cell_5/sub:z:0)gru_1/while/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚Ь
gru_1/while/gru_cell_5/add_3AddV2 gru_1/while/gru_cell_5/mul_1:z:0 gru_1/while/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚x
6gru_1/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Г
0gru_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_1_while_placeholder_1?gru_1/while/TensorArrayV2Write/TensorListSetItem/index:output:0 gru_1/while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥S
gru_1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_1/while/addAddV2gru_1_while_placeholdergru_1/while/add/y:output:0*
T0*
_output_shapes
: U
gru_1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_1/while/add_1AddV2$gru_1_while_gru_1_while_loop_countergru_1/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_1/while/IdentityIdentitygru_1/while/add_1:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: В
gru_1/while/Identity_1Identity*gru_1_while_gru_1_while_maximum_iterations^gru_1/while/NoOp*
T0*
_output_shapes
: k
gru_1/while/Identity_2Identitygru_1/while/add:z:0^gru_1/while/NoOp*
T0*
_output_shapes
: Ш
gru_1/while/Identity_3Identity@gru_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_1/while/NoOp*
T0*
_output_shapes
: К
gru_1/while/Identity_4Identity gru_1/while/gru_cell_5/add_3:z:0^gru_1/while/NoOp*
T0*(
_output_shapes
:         ╚┌
gru_1/while/NoOpNoOp-^gru_1/while/gru_cell_5/MatMul/ReadVariableOp/^gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp&^gru_1/while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_1_while_gru_1_strided_slice_1#gru_1_while_gru_1_strided_slice_1_0"t
7gru_1_while_gru_cell_5_matmul_1_readvariableop_resource9gru_1_while_gru_cell_5_matmul_1_readvariableop_resource_0"p
5gru_1_while_gru_cell_5_matmul_readvariableop_resource7gru_1_while_gru_cell_5_matmul_readvariableop_resource_0"b
.gru_1_while_gru_cell_5_readvariableop_resource0gru_1_while_gru_cell_5_readvariableop_resource_0"5
gru_1_while_identitygru_1/while/Identity:output:0"9
gru_1_while_identity_1gru_1/while/Identity_1:output:0"9
gru_1_while_identity_2gru_1/while/Identity_2:output:0"9
gru_1_while_identity_3gru_1/while/Identity_3:output:0"9
gru_1_while_identity_4gru_1/while/Identity_4:output:0"└
]gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_gru_1_while_tensorarrayv2read_tensorlistgetitem_gru_1_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2\
,gru_1/while/gru_cell_5/MatMul/ReadVariableOp,gru_1/while/gru_cell_5/MatMul/ReadVariableOp2`
.gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp.gru_1/while/gru_cell_5/MatMul_1/ReadVariableOp2N
%gru_1/while/gru_cell_5/ReadVariableOp%gru_1/while/gru_cell_5/ReadVariableOp:P L

_output_shapes
: 
2
_user_specified_namegru_1/while/loop_counter:VR

_output_shapes
: 
8
_user_specified_name gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
Ў	
ж
gru_1_while_cond_19998288(
$gru_1_while_gru_1_while_loop_counter.
*gru_1_while_gru_1_while_maximum_iterations
gru_1_while_placeholder
gru_1_while_placeholder_1
gru_1_while_placeholder_2*
&gru_1_while_less_gru_1_strided_slice_1B
>gru_1_while_gru_1_while_cond_19998288___redundant_placeholder0B
>gru_1_while_gru_1_while_cond_19998288___redundant_placeholder1B
>gru_1_while_gru_1_while_cond_19998288___redundant_placeholder2B
>gru_1_while_gru_1_while_cond_19998288___redundant_placeholder3
gru_1_while_identity
z
gru_1/while/LessLessgru_1_while_placeholder&gru_1_while_less_gru_1_strided_slice_1*
T0*
_output_shapes
: W
gru_1/while/IdentityIdentitygru_1/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_1_while_identitygru_1/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::P L

_output_shapes
: 
2
_user_specified_namegru_1/while/loop_counter:VR

_output_shapes
: 
8
_user_specified_name gru_1/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
в

f
G__inference_dropout_1_layer_call_and_return_conditional_losses_20000074

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╚Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         ╚b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         ╚"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
ж
▄
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_19996946

inputs

states*
readvariableop_resource:	╪2
matmul_readvariableop_resource:
╚╪4
 matmul_1_readvariableop_resource:
╚╪
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	╪*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         ╪Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         ╪Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         ╚N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         ╚c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         ╚R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         ╚^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         ╚Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         ╚J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:         ╚T
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:         ╚J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         ╚\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:         ╚W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         ╚Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         ╚:         ╚: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ╚
 
_user_specified_namestates
╤>
Е
while_body_19999652
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_5_readvariableop_resource_0:	╪E
1while_gru_cell_5_matmul_readvariableop_resource_0:
╚╪G
3while_gru_cell_5_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_5_readvariableop_resource:	╪C
/while_gru_cell_5_matmul_readvariableop_resource:
╚╪E
1while_gru_cell_5_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_5/MatMul/ReadVariableOpв(while/gru_cell_5/MatMul_1/ReadVariableOpвwhile/gru_cell_5/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0Л
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЪ
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0╢
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
м
║
&__inference_gru_layer_call_fn_19998716
inputs_0
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_19996541}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
╚
┤
while_cond_19997318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19997318___redundant_placeholder06
2while_while_cond_19997318___redundant_placeholder16
2while_while_cond_19997318___redundant_placeholder26
2while_while_cond_19997318___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
ф	
╥
-__inference_sequential_layer_call_fn_19998067

inputs
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
	unknown_2:	╪
	unknown_3:
╚╪
	unknown_4:
╚╪
	unknown_5:	╚
	unknown_6:
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_19997865o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╤>
Е
while_body_19997319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_5_readvariableop_resource_0:	╪E
1while_gru_cell_5_matmul_readvariableop_resource_0:
╚╪G
3while_gru_cell_5_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_5_readvariableop_resource:	╪C
/while_gru_cell_5_matmul_readvariableop_resource:
╚╪E
1while_gru_cell_5_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_5/MatMul/ReadVariableOpв(while/gru_cell_5/MatMul_1/ReadVariableOpвwhile/gru_cell_5/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0Л
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЪ
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0╢
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
▐
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_19997783

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╚\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╚"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
в
█
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_19996464

inputs

states*
readvariableop_resource:	╪1
matmul_readvariableop_resource:	╪4
 matmul_1_readvariableop_resource:
╚╪
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	╪*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪i
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         ╪Z
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪m
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         ╪Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         ╚N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         ╚c
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         ╚R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         ╚^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         ╚Z
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         ╚J
ReluRelu	add_2:z:0*
T0*(
_output_shapes
:         ╚T
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:         ╚J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         ╚\
mul_2Mulsub:z:0Relu:activations:0*
T0*(
_output_shapes
:         ╚W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         ╚Y
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         ╚Й
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ╚: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ╚
 
_user_specified_namestates
▒=
Г
while_body_19999119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_4_readvariableop_resource_0:	╪D
1while_gru_cell_4_matmul_readvariableop_resource_0:	╪G
3while_gru_cell_4_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_4_readvariableop_resource:	╪B
/while_gru_cell_4_matmul_readvariableop_resource:	╪E
1while_gru_cell_4_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_4/MatMul/ReadVariableOpв(while/gru_cell_4/MatMul_1/ReadVariableOpвwhile/gru_cell_4/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Л
while/gru_cell_4/ReadVariableOpReadVariableOp*while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_4/unstackUnpack'while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЩ
&while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0╢
while/gru_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_4/BiasAddBiasAdd!while/gru_cell_4/MatMul:product:0!while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_4/splitSplit)while/gru_cell_4/split/split_dim:output:0!while/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_4/MatMul_1MatMulwhile_placeholder_20while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_4/BiasAdd_1BiasAdd#while/gru_cell_4/MatMul_1:product:0!while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_4/split_1SplitV#while/gru_cell_4/BiasAdd_1:output:0while/gru_cell_4/Const:output:0+while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_4/addAddV2while/gru_cell_4/split:output:0!while/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_4/SigmoidSigmoidwhile/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_4/add_1AddV2while/gru_cell_4/split:output:1!while/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_4/Sigmoid_1Sigmoidwhile/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_4/mulMulwhile/gru_cell_4/Sigmoid_1:y:0!while/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_4/add_2AddV2while/gru_cell_4/split:output:2while/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_4/ReluReluwhile/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_4/mul_1Mulwhile/gru_cell_4/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_4/subSubwhile/gru_cell_4/sub/x:output:0while/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_4/mul_2Mulwhile/gru_cell_4/sub:z:0#while/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_4/add_3AddV2while/gru_cell_4/mul_1:z:0while/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚├
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_4/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_4/MatMul/ReadVariableOp)^while/gru_cell_4/MatMul_1/ReadVariableOp ^while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_4_matmul_1_readvariableop_resource3while_gru_cell_4_matmul_1_readvariableop_resource_0"d
/while_gru_cell_4_matmul_readvariableop_resource1while_gru_cell_4_matmul_readvariableop_resource_0"V
(while_gru_cell_4_readvariableop_resource*while_gru_cell_4_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_4/MatMul/ReadVariableOp&while/gru_cell_4/MatMul/ReadVariableOp2T
(while/gru_cell_4/MatMul_1/ReadVariableOp(while/gru_cell_4/MatMul_1/ReadVariableOp2B
while/gru_cell_4/ReadVariableOpwhile/gru_cell_4/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
ю 
╛
while_body_19996619
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_gru_cell_4_19996641_0:	╪.
while_gru_cell_4_19996643_0:	╪/
while_gru_cell_4_19996645_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_gru_cell_4_19996641:	╪,
while_gru_cell_4_19996643:	╪-
while_gru_cell_4_19996645:
╚╪Ив(while/gru_cell_4/StatefulPartitionedCallИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ж
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0Й
(while/gru_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_4_19996641_0while_gru_cell_4_19996643_0while_gru_cell_4_19996645_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_19996606┌
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder1while/gru_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: П
while/Identity_4Identity1while/gru_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ╚w

while/NoOpNoOp)^while/gru_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "8
while_gru_cell_4_19996641while_gru_cell_4_19996641_0"8
while_gru_cell_4_19996643while_gru_cell_4_19996643_0"8
while_gru_cell_4_19996645while_gru_cell_4_19996645_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2T
(while/gru_cell_4/StatefulPartitionedCall(while/gru_cell_4/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
БM
В
A__inference_gru_layer_call_and_return_conditional_losses_19997233

inputs5
"gru_cell_4_readvariableop_resource:	╪<
)gru_cell_4_matmul_readvariableop_resource:	╪?
+gru_cell_4_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_4/MatMul/ReadVariableOpв"gru_cell_4/MatMul_1/ReadVariableOpвgru_cell_4/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask}
gru_cell_4/ReadVariableOpReadVariableOp"gru_cell_4_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_4/unstackUnpack!gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЛ
 gru_cell_4/MatMul/ReadVariableOpReadVariableOp)gru_cell_4_matmul_readvariableop_resource*
_output_shapes
:	╪*
dtype0Т
gru_cell_4/MatMulMatMulstrided_slice_2:output:0(gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_4/BiasAddBiasAddgru_cell_4/MatMul:product:0gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_4/splitSplit#gru_cell_4/split/split_dim:output:0gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_4_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_4/MatMul_1MatMulzeros:output:0*gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_4/BiasAdd_1BiasAddgru_cell_4/MatMul_1:product:0gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_4/split_1SplitVgru_cell_4/BiasAdd_1:output:0gru_cell_4/Const:output:0%gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_4/addAddV2gru_cell_4/split:output:0gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_4/SigmoidSigmoidgru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_4/add_1AddV2gru_cell_4/split:output:1gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_4/Sigmoid_1Sigmoidgru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_4/mulMulgru_cell_4/Sigmoid_1:y:0gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_4/add_2AddV2gru_cell_4/split:output:2gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_4/ReluRelugru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_4/mul_1Mulgru_cell_4/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_4/subSubgru_cell_4/sub/x:output:0gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_4/mul_2Mulgru_cell_4/sub:z:0gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_4/add_3AddV2gru_cell_4/mul_1:z:0gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_4_readvariableop_resource)gru_cell_4_matmul_readvariableop_resource+gru_cell_4_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19997144*
condR
while_cond_19997143*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ├
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    c
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ╚▓
NoOpNoOp!^gru_cell_4/MatMul/ReadVariableOp#^gru_cell_4/MatMul_1/ReadVariableOp^gru_cell_4/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2D
 gru_cell_4/MatMul/ReadVariableOp gru_cell_4/MatMul/ReadVariableOp2H
"gru_cell_4/MatMul_1/ReadVariableOp"gru_cell_4/MatMul_1/ReadVariableOp26
gru_cell_4/ReadVariableOpgru_cell_4/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
║B
√
gru_while_body_19998131$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0A
.gru_while_gru_cell_4_readvariableop_resource_0:	╪H
5gru_while_gru_cell_4_matmul_readvariableop_resource_0:	╪K
7gru_while_gru_cell_4_matmul_1_readvariableop_resource_0:
╚╪
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor?
,gru_while_gru_cell_4_readvariableop_resource:	╪F
3gru_while_gru_cell_4_matmul_readvariableop_resource:	╪I
5gru_while_gru_cell_4_matmul_1_readvariableop_resource:
╚╪Ив*gru/while/gru_cell_4/MatMul/ReadVariableOpв,gru/while/gru_cell_4/MatMul_1/ReadVariableOpв#gru/while/gru_cell_4/ReadVariableOpМ
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ║
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0У
#gru/while/gru_cell_4/ReadVariableOpReadVariableOp.gru_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Л
gru/while/gru_cell_4/unstackUnpack+gru/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numб
*gru/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp5gru_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0┬
gru/while/gru_cell_4/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:02gru/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪и
gru/while/gru_cell_4/BiasAddBiasAdd%gru/while/gru_cell_4/MatMul:product:0%gru/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪o
$gru/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         х
gru/while/gru_cell_4/splitSplit-gru/while/gru_cell_4/split/split_dim:output:0%gru/while/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitж
,gru/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp7gru_while_gru_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0й
gru/while/gru_cell_4/MatMul_1MatMulgru_while_placeholder_24gru/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪м
gru/while/gru_cell_4/BiasAdd_1BiasAdd'gru/while/gru_cell_4/MatMul_1:product:0%gru/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪o
gru/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       q
&gru/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Э
gru/while/gru_cell_4/split_1SplitV'gru/while/gru_cell_4/BiasAdd_1:output:0#gru/while/gru_cell_4/Const:output:0/gru/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitа
gru/while/gru_cell_4/addAddV2#gru/while/gru_cell_4/split:output:0%gru/while/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚x
gru/while/gru_cell_4/SigmoidSigmoidgru/while/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚в
gru/while/gru_cell_4/add_1AddV2#gru/while/gru_cell_4/split:output:1%gru/while/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚|
gru/while/gru_cell_4/Sigmoid_1Sigmoidgru/while/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚Э
gru/while/gru_cell_4/mulMul"gru/while/gru_cell_4/Sigmoid_1:y:0%gru/while/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚Щ
gru/while/gru_cell_4/add_2AddV2#gru/while/gru_cell_4/split:output:2gru/while/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚t
gru/while/gru_cell_4/ReluRelugru/while/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚П
gru/while/gru_cell_4/mul_1Mul gru/while/gru_cell_4/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:         ╚_
gru/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
gru/while/gru_cell_4/subSub#gru/while/gru_cell_4/sub/x:output:0 gru/while/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚Ы
gru/while/gru_cell_4/mul_2Mulgru/while/gru_cell_4/sub:z:0'gru/while/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚Ц
gru/while/gru_cell_4/add_3AddV2gru/while/gru_cell_4/mul_1:z:0gru/while/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚╙
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥Q
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: S
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: e
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: z
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: e
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: Т
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: Д
gru/while/Identity_4Identitygru/while/gru_cell_4/add_3:z:0^gru/while/NoOp*
T0*(
_output_shapes
:         ╚╥
gru/while/NoOpNoOp+^gru/while/gru_cell_4/MatMul/ReadVariableOp-^gru/while/gru_cell_4/MatMul_1/ReadVariableOp$^gru/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "p
5gru_while_gru_cell_4_matmul_1_readvariableop_resource7gru_while_gru_cell_4_matmul_1_readvariableop_resource_0"l
3gru_while_gru_cell_4_matmul_readvariableop_resource5gru_while_gru_cell_4_matmul_readvariableop_resource_0"^
,gru_while_gru_cell_4_readvariableop_resource.gru_while_gru_cell_4_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"╕
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2X
*gru/while/gru_cell_4/MatMul/ReadVariableOp*gru/while/gru_cell_4/MatMul/ReadVariableOp2\
,gru/while/gru_cell_4/MatMul_1/ReadVariableOp,gru/while/gru_cell_4/MatMul_1/ReadVariableOp2J
#gru/while/gru_cell_4/ReadVariableOp#gru/while/gru_cell_4/ReadVariableOp:N J

_output_shapes
: 
0
_user_specified_namegru/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namegru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
▓5
Й
C__inference_gru_1_layer_call_and_return_conditional_losses_19997025

inputs&
gru_cell_5_19996947:	╪'
gru_cell_5_19996949:
╚╪'
gru_cell_5_19996951:
╚╪
identityИв"gru_cell_5/StatefulPartitionedCallвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:                  ╚R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_mask╬
"gru_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_5_19996947gru_cell_5_19996949gru_cell_5_19996951*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_19996946n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Д
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_5_19996947gru_cell_5_19996949gru_cell_5_19996951*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19996960*
condR
while_cond_19996959*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╫
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ╚s
NoOpNoOp#^gru_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ╚: : : 2H
"gru_cell_5/StatefulPartitionedCall"gru_cell_5/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  ╚
 
_user_specified_nameinputs
╚
┤
while_cond_19997513
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19997513___redundant_placeholder06
2while_while_cond_19997513___redundant_placeholder16
2while_while_cond_19997513___redundant_placeholder26
2while_while_cond_19997513___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
▐
e
G__inference_dropout_1_layer_call_and_return_conditional_losses_20000079

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╚\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╚"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
ПN
Е
C__inference_gru_1_layer_call_and_return_conditional_losses_19997409

inputs5
"gru_cell_5_readvariableop_resource:	╪=
)gru_cell_5_matmul_readvariableop_resource:
╚╪?
+gru_cell_5_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_5/MatMul/ReadVariableOpв"gru_cell_5/MatMul_1/ReadVariableOpвgru_cell_5/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         ╚R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_mask}
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numМ
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0Т
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19997319*
condR
while_cond_19997318*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╫
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ╚▓
NoOpNoOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ╚: : : 2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
╝

d
E__inference_dropout_layer_call_and_return_conditional_losses_19997253

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ╚Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧С
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ╚*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?л
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╚T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ш
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:         ╚f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:         ╚"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
ь
c
E__inference_dropout_layer_call_and_return_conditional_losses_19999388

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:         ╚`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ╚"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
╚
┤
while_cond_19999496
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19999496___redundant_placeholder06
2while_while_cond_19999496___redundant_placeholder16
2while_while_cond_19999496___redundant_placeholder26
2while_while_cond_19999496___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
└

▐
-__inference_gru_cell_5_layer_call_fn_20000232

inputs
states_0
unknown:	╪
	unknown_0:
╚╪
	unknown_1:
╚╪
identity

identity_1ИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_19996946p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         ╚:         ╚: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╚
"
_user_specified_name
states_0
┴	
╬
&__inference_signature_wrapper_19998025
	gru_input
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
	unknown_2:	╪
	unknown_3:
╚╪
	unknown_4:
╚╪
	unknown_5:	╚
	unknown_6:
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCall	gru_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_19996397o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
+
_output_shapes
:         
#
_user_specified_name	gru_input
╤>
Е
while_body_19999807
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
*while_gru_cell_5_readvariableop_resource_0:	╪E
1while_gru_cell_5_matmul_readvariableop_resource_0:
╚╪G
3while_gru_cell_5_matmul_1_readvariableop_resource_0:
╚╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
(while_gru_cell_5_readvariableop_resource:	╪C
/while_gru_cell_5_matmul_readvariableop_resource:
╚╪E
1while_gru_cell_5_matmul_1_readvariableop_resource:
╚╪Ив&while/gru_cell_5/MatMul/ReadVariableOpв(while/gru_cell_5/MatMul_1/ReadVariableOpвwhile/gru_cell_5/ReadVariableOpИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   з
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ╚*
element_dtype0Л
while/gru_cell_5/ReadVariableOpReadVariableOp*while_gru_cell_5_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Г
while/gru_cell_5/unstackUnpack'while/gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numЪ
&while/gru_cell_5/MatMul/ReadVariableOpReadVariableOp1while_gru_cell_5_matmul_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0╢
while/gru_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0.while/gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪Ь
while/gru_cell_5/BiasAddBiasAdd!while/gru_cell_5/MatMul:product:0!while/gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪k
 while/gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ┘
while/gru_cell_5/splitSplit)while/gru_cell_5/split/split_dim:output:0!while/gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitЮ
(while/gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp3while_gru_cell_5_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0Э
while/gru_cell_5/MatMul_1MatMulwhile_placeholder_20while/gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪а
while/gru_cell_5/BiasAdd_1BiasAdd#while/gru_cell_5/MatMul_1:product:0!while/gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪k
while/gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       m
"while/gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Н
while/gru_cell_5/split_1SplitV#while/gru_cell_5/BiasAdd_1:output:0while/gru_cell_5/Const:output:0+while/gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitФ
while/gru_cell_5/addAddV2while/gru_cell_5/split:output:0!while/gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚p
while/gru_cell_5/SigmoidSigmoidwhile/gru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Ц
while/gru_cell_5/add_1AddV2while/gru_cell_5/split:output:1!while/gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚t
while/gru_cell_5/Sigmoid_1Sigmoidwhile/gru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚С
while/gru_cell_5/mulMulwhile/gru_cell_5/Sigmoid_1:y:0!while/gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚Н
while/gru_cell_5/add_2AddV2while/gru_cell_5/split:output:2while/gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚l
while/gru_cell_5/ReluReluwhile/gru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚Г
while/gru_cell_5/mul_1Mulwhile/gru_cell_5/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         ╚[
while/gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Н
while/gru_cell_5/subSubwhile/gru_cell_5/sub/x:output:0while/gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚П
while/gru_cell_5/mul_2Mulwhile/gru_cell_5/sub:z:0#while/gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚К
while/gru_cell_5/add_3AddV2while/gru_cell_5/mul_1:z:0while/gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ы
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/gru_cell_5/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ж
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/gru_cell_5/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:         ╚┬

while/NoOpNoOp'^while/gru_cell_5/MatMul/ReadVariableOp)^while/gru_cell_5/MatMul_1/ReadVariableOp ^while/gru_cell_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "h
1while_gru_cell_5_matmul_1_readvariableop_resource3while_gru_cell_5_matmul_1_readvariableop_resource_0"d
/while_gru_cell_5_matmul_readvariableop_resource1while_gru_cell_5_matmul_readvariableop_resource_0"V
(while_gru_cell_5_readvariableop_resource*while_gru_cell_5_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2P
&while/gru_cell_5/MatMul/ReadVariableOp&while/gru_cell_5/MatMul/ReadVariableOp2T
(while/gru_cell_5/MatMul_1/ReadVariableOp(while/gru_cell_5/MatMul_1/ReadVariableOp2B
while/gru_cell_5/ReadVariableOpwhile/gru_cell_5/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: 
╚
┤
while_cond_19996959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19996959___redundant_placeholder06
2while_while_cond_19996959___redundant_placeholder16
2while_while_cond_19996959___redundant_placeholder26
2while_while_cond_19996959___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
Ё
╪
H__inference_sequential_layer_call_and_return_conditional_losses_19997791
	gru_input
gru_19997604:	╪
gru_19997606:	╪ 
gru_19997608:
╚╪!
gru_1_19997772:	╪"
gru_1_19997774:
╚╪"
gru_1_19997776:
╚╪!
dense_19997785:	╚
dense_19997787:
identityИвdense/StatefulPartitionedCallвgru/StatefulPartitionedCallвgru_1/StatefulPartitionedCall·
gru/StatefulPartitionedCallStatefulPartitionedCall	gru_inputgru_19997604gru_19997606gru_19997608*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_gru_layer_call_and_return_conditional_losses_19997603█
dropout/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_19997615Ч
gru_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0gru_1_19997772gru_1_19997774gru_1_19997776*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_1_layer_call_and_return_conditional_losses_19997771▌
dropout_1/PartitionedCallPartitionedCall&gru_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_dropout_1_layer_call_and_return_conditional_losses_19997783Ж
dense/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_19997785dense_19997787*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_19997441u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         д
NoOpNoOp^dense/StatefulPartitionedCall^gru/StatefulPartitionedCall^gru_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall:V R
+
_output_shapes
:         
#
_user_specified_name	gru_input
╚
┤
while_cond_19997143
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19997143___redundant_placeholder06
2while_while_cond_19997143___redundant_placeholder16
2while_while_cond_19997143___redundant_placeholder26
2while_while_cond_19997143___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
╚
┤
while_cond_19996815
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_19996815___redundant_placeholder06
2while_while_cond_19996815___redundant_placeholder16
2while_while_cond_19996815___redundant_placeholder26
2while_while_cond_19996815___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : :         ╚: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
:
в

f
G__inference_dropout_1_layer_call_and_return_conditional_losses_19997429

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╚Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         ╚b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         ╚"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
ПN
Е
C__inference_gru_1_layer_call_and_return_conditional_losses_20000052

inputs5
"gru_cell_5_readvariableop_resource:	╪=
)gru_cell_5_matmul_readvariableop_resource:
╚╪?
+gru_cell_5_matmul_1_readvariableop_resource:
╚╪
identityИв gru_cell_5/MatMul/ReadVariableOpв"gru_cell_5/MatMul_1/ReadVariableOpвgru_cell_5/ReadVariableOpвwhileI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╚s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╚c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         ╚R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::э╧_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_mask}
gru_cell_5/ReadVariableOpReadVariableOp"gru_cell_5_readvariableop_resource*
_output_shapes
:	╪*
dtype0w
gru_cell_5/unstackUnpack!gru_cell_5/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numМ
 gru_cell_5/MatMul/ReadVariableOpReadVariableOp)gru_cell_5_matmul_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0Т
gru_cell_5/MatMulMatMulstrided_slice_2:output:0(gru_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪К
gru_cell_5/BiasAddBiasAddgru_cell_5/MatMul:product:0gru_cell_5/unstack:output:0*
T0*(
_output_shapes
:         ╪e
gru_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╟
gru_cell_5/splitSplit#gru_cell_5/split/split_dim:output:0gru_cell_5/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitР
"gru_cell_5/MatMul_1/ReadVariableOpReadVariableOp+gru_cell_5_matmul_1_readvariableop_resource* 
_output_shapes
:
╚╪*
dtype0М
gru_cell_5/MatMul_1MatMulzeros:output:0*gru_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪О
gru_cell_5/BiasAdd_1BiasAddgru_cell_5/MatMul_1:product:0gru_cell_5/unstack:output:1*
T0*(
_output_shapes
:         ╪e
gru_cell_5/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       g
gru_cell_5/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ї
gru_cell_5/split_1SplitVgru_cell_5/BiasAdd_1:output:0gru_cell_5/Const:output:0%gru_cell_5/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitВ
gru_cell_5/addAddV2gru_cell_5/split:output:0gru_cell_5/split_1:output:0*
T0*(
_output_shapes
:         ╚d
gru_cell_5/SigmoidSigmoidgru_cell_5/add:z:0*
T0*(
_output_shapes
:         ╚Д
gru_cell_5/add_1AddV2gru_cell_5/split:output:1gru_cell_5/split_1:output:1*
T0*(
_output_shapes
:         ╚h
gru_cell_5/Sigmoid_1Sigmoidgru_cell_5/add_1:z:0*
T0*(
_output_shapes
:         ╚
gru_cell_5/mulMulgru_cell_5/Sigmoid_1:y:0gru_cell_5/split_1:output:2*
T0*(
_output_shapes
:         ╚{
gru_cell_5/add_2AddV2gru_cell_5/split:output:2gru_cell_5/mul:z:0*
T0*(
_output_shapes
:         ╚`
gru_cell_5/ReluRelugru_cell_5/add_2:z:0*
T0*(
_output_shapes
:         ╚r
gru_cell_5/mul_1Mulgru_cell_5/Sigmoid:y:0zeros:output:0*
T0*(
_output_shapes
:         ╚U
gru_cell_5/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?{
gru_cell_5/subSubgru_cell_5/sub/x:output:0gru_cell_5/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚}
gru_cell_5/mul_2Mulgru_cell_5/sub:z:0gru_cell_5/Relu:activations:0*
T0*(
_output_shapes
:         ╚x
gru_cell_5/add_3AddV2gru_cell_5/mul_1:z:0gru_cell_5/mul_2:z:0*
T0*(
_output_shapes
:         ╚n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :┼
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щш╥F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ┴
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0"gru_cell_5_readvariableop_resource)gru_cell_5_matmul_readvariableop_resource+gru_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*:
_output_shapes(
&: : : : :         ╚: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_19999962*
condR
while_cond_19999961*9
output_shapes(
&: : : : :         ╚: : : : : *
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ╚   ╫
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╚*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╚*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ч
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╚[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    h
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*(
_output_shapes
:         ╚▓
NoOpNoOp!^gru_cell_5/MatMul/ReadVariableOp#^gru_cell_5/MatMul_1/ReadVariableOp^gru_cell_5/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ╚: : : 2D
 gru_cell_5/MatMul/ReadVariableOp gru_cell_5/MatMul/ReadVariableOp2H
"gru_cell_5/MatMul_1/ReadVariableOp"gru_cell_5/MatMul_1/ReadVariableOp26
gru_cell_5/ReadVariableOpgru_cell_5/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
╤.
├
$__inference__traced_restore_20000433
file_prefix0
assignvariableop_dense_kernel:	╚+
assignvariableop_1_dense_bias:9
&assignvariableop_2_gru_gru_cell_kernel:	╪D
0assignvariableop_3_gru_gru_cell_recurrent_kernel:
╚╪7
$assignvariableop_4_gru_gru_cell_bias:	╪>
*assignvariableop_5_gru_1_gru_cell_1_kernel:
╚╪H
4assignvariableop_6_gru_1_gru_cell_1_recurrent_kernel:
╚╪;
(assignvariableop_7_gru_1_gru_cell_1_bias:	╪"
assignvariableop_8_total: "
assignvariableop_9_count: 
identity_11ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╫
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¤
valueєBЁB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ╒
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:░
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╜
AssignVariableOp_2AssignVariableOp&assignvariableop_2_gru_gru_cell_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╟
AssignVariableOp_3AssignVariableOp0assignvariableop_3_gru_gru_cell_recurrent_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_4AssignVariableOp$assignvariableop_4_gru_gru_cell_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_5AssignVariableOp*assignvariableop_5_gru_1_gru_cell_1_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╦
AssignVariableOp_6AssignVariableOp4assignvariableop_6_gru_1_gru_cell_1_recurrent_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_7AssignVariableOp(assignvariableop_7_gru_1_gru_cell_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 л
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: Ш
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╩	
ї
C__inference_dense_layer_call_and_return_conditional_losses_19997441

inputs1
matmul_readvariableop_resource:	╚-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╚: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
╜

▌
-__inference_gru_cell_4_layer_call_fn_20000126

inputs
states_0
unknown:	╪
	unknown_0:	╪
	unknown_1:
╚╪
identity

identity_1ИвStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         ╚:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_19996606p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:         :         ╚: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╚
"
_user_specified_name
states_0
Б
╗
(__inference_gru_1_layer_call_fn_19999421

inputs
unknown:	╪
	unknown_0:
╚╪
	unknown_1:
╚╪
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ╚*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_gru_1_layer_call_and_return_conditional_losses_19997409p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╚`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ╚: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╚
 
_user_specified_nameinputs
║B
√
gru_while_body_19998457$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0A
.gru_while_gru_cell_4_readvariableop_resource_0:	╪H
5gru_while_gru_cell_4_matmul_readvariableop_resource_0:	╪K
7gru_while_gru_cell_4_matmul_1_readvariableop_resource_0:
╚╪
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor?
,gru_while_gru_cell_4_readvariableop_resource:	╪F
3gru_while_gru_cell_4_matmul_readvariableop_resource:	╪I
5gru_while_gru_cell_4_matmul_1_readvariableop_resource:
╚╪Ив*gru/while/gru_cell_4/MatMul/ReadVariableOpв,gru/while/gru_cell_4/MatMul_1/ReadVariableOpв#gru/while/gru_cell_4/ReadVariableOpМ
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ║
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0У
#gru/while/gru_cell_4/ReadVariableOpReadVariableOp.gru_while_gru_cell_4_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0Л
gru/while/gru_cell_4/unstackUnpack+gru/while/gru_cell_4/ReadVariableOp:value:0*
T0*"
_output_shapes
:╪:╪*	
numб
*gru/while/gru_cell_4/MatMul/ReadVariableOpReadVariableOp5gru_while_gru_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	╪*
dtype0┬
gru/while/gru_cell_4/MatMulMatMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:02gru/while/gru_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪и
gru/while/gru_cell_4/BiasAddBiasAdd%gru/while/gru_cell_4/MatMul:product:0%gru/while/gru_cell_4/unstack:output:0*
T0*(
_output_shapes
:         ╪o
$gru/while/gru_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         х
gru/while/gru_cell_4/splitSplit-gru/while/gru_cell_4/split/split_dim:output:0%gru/while/gru_cell_4/BiasAdd:output:0*
T0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitж
,gru/while/gru_cell_4/MatMul_1/ReadVariableOpReadVariableOp7gru_while_gru_cell_4_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╚╪*
dtype0й
gru/while/gru_cell_4/MatMul_1MatMulgru_while_placeholder_24gru/while/gru_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪м
gru/while/gru_cell_4/BiasAdd_1BiasAdd'gru/while/gru_cell_4/MatMul_1:product:0%gru/while/gru_cell_4/unstack:output:1*
T0*(
_output_shapes
:         ╪o
gru/while/gru_cell_4/ConstConst*
_output_shapes
:*
dtype0*!
valueB"╚   ╚       q
&gru/while/gru_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Э
gru/while/gru_cell_4/split_1SplitV'gru/while/gru_cell_4/BiasAdd_1:output:0#gru/while/gru_cell_4/Const:output:0/gru/while/gru_cell_4/split_1/split_dim:output:0*
T0*

Tlen0*P
_output_shapes>
<:         ╚:         ╚:         ╚*
	num_splitа
gru/while/gru_cell_4/addAddV2#gru/while/gru_cell_4/split:output:0%gru/while/gru_cell_4/split_1:output:0*
T0*(
_output_shapes
:         ╚x
gru/while/gru_cell_4/SigmoidSigmoidgru/while/gru_cell_4/add:z:0*
T0*(
_output_shapes
:         ╚в
gru/while/gru_cell_4/add_1AddV2#gru/while/gru_cell_4/split:output:1%gru/while/gru_cell_4/split_1:output:1*
T0*(
_output_shapes
:         ╚|
gru/while/gru_cell_4/Sigmoid_1Sigmoidgru/while/gru_cell_4/add_1:z:0*
T0*(
_output_shapes
:         ╚Э
gru/while/gru_cell_4/mulMul"gru/while/gru_cell_4/Sigmoid_1:y:0%gru/while/gru_cell_4/split_1:output:2*
T0*(
_output_shapes
:         ╚Щ
gru/while/gru_cell_4/add_2AddV2#gru/while/gru_cell_4/split:output:2gru/while/gru_cell_4/mul:z:0*
T0*(
_output_shapes
:         ╚t
gru/while/gru_cell_4/ReluRelugru/while/gru_cell_4/add_2:z:0*
T0*(
_output_shapes
:         ╚П
gru/while/gru_cell_4/mul_1Mul gru/while/gru_cell_4/Sigmoid:y:0gru_while_placeholder_2*
T0*(
_output_shapes
:         ╚_
gru/while/gru_cell_4/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
gru/while/gru_cell_4/subSub#gru/while/gru_cell_4/sub/x:output:0 gru/while/gru_cell_4/Sigmoid:y:0*
T0*(
_output_shapes
:         ╚Ы
gru/while/gru_cell_4/mul_2Mulgru/while/gru_cell_4/sub:z:0'gru/while/gru_cell_4/Relu:activations:0*
T0*(
_output_shapes
:         ╚Ц
gru/while/gru_cell_4/add_3AddV2gru/while/gru_cell_4/mul_1:z:0gru/while/gru_cell_4/mul_2:z:0*
T0*(
_output_shapes
:         ╚╙
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell_4/add_3:z:0*
_output_shapes
: *
element_dtype0:щш╥Q
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: S
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: e
gru/while/IdentityIdentitygru/while/add_1:z:0^gru/while/NoOp*
T0*
_output_shapes
: z
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations^gru/while/NoOp*
T0*
_output_shapes
: e
gru/while/Identity_2Identitygru/while/add:z:0^gru/while/NoOp*
T0*
_output_shapes
: Т
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru/while/NoOp*
T0*
_output_shapes
: Д
gru/while/Identity_4Identitygru/while/gru_cell_4/add_3:z:0^gru/while/NoOp*
T0*(
_output_shapes
:         ╚╥
gru/while/NoOpNoOp+^gru/while/gru_cell_4/MatMul/ReadVariableOp-^gru/while/gru_cell_4/MatMul_1/ReadVariableOp$^gru/while/gru_cell_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "p
5gru_while_gru_cell_4_matmul_1_readvariableop_resource7gru_while_gru_cell_4_matmul_1_readvariableop_resource_0"l
3gru_while_gru_cell_4_matmul_readvariableop_resource5gru_while_gru_cell_4_matmul_readvariableop_resource_0"^
,gru_while_gru_cell_4_readvariableop_resource.gru_while_gru_cell_4_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"╕
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :         ╚: : : : : 2X
*gru/while/gru_cell_4/MatMul/ReadVariableOp*gru/while/gru_cell_4/MatMul/ReadVariableOp2\
,gru/while/gru_cell_4/MatMul_1/ReadVariableOp,gru/while/gru_cell_4/MatMul_1/ReadVariableOp2J
#gru/while/gru_cell_4/ReadVariableOp#gru/while/gru_cell_4/ReadVariableOp:N J

_output_shapes
: 
0
_user_specified_namegru/while/loop_counter:TP

_output_shapes
: 
6
_user_specified_namegru/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╚:

_output_shapes
: :

_output_shapes
: "є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*░
serving_defaultЬ
C
	gru_input6
serving_default_gru_input:0         9
dense0
StatefulPartitionedCall:0         tensorflow/serving/predict:°У
Ъ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_sequential
 
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec
#_self_saveable_object_factories"
_tf_keras_rnn_layer
с
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
 _random_generator
#!_self_saveable_object_factories"
_tf_keras_layer
 
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(_random_generator
)cell
*
state_spec
#+_self_saveable_object_factories"
_tf_keras_rnn_layer
с
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator
#3_self_saveable_object_factories"
_tf_keras_layer
р
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

:kernel
;bias
#<_self_saveable_object_factories"
_tf_keras_layer
X
=0
>1
?2
@3
A4
B5
:6
;7"
trackable_list_wrapper
X
=0
>1
?2
@3
A4
B5
:6
;7"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
▀
Htrace_0
Itrace_1
Jtrace_2
Ktrace_32Ї
-__inference_sequential_layer_call_fn_19997838
-__inference_sequential_layer_call_fn_19997884
-__inference_sequential_layer_call_fn_19998046
-__inference_sequential_layer_call_fn_19998067╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zHtrace_0zItrace_1zJtrace_2zKtrace_3
╦
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32р
H__inference_sequential_layer_call_and_return_conditional_losses_19997448
H__inference_sequential_layer_call_and_return_conditional_losses_19997791
H__inference_sequential_layer_call_and_return_conditional_losses_19998393
H__inference_sequential_layer_call_and_return_conditional_losses_19998705╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
╨B═
#__inference__wrapped_model_19996397	gru_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
	optimizer
,
Pserving_default"
signature_map
 "
trackable_dict_wrapper
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

Qstates
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╪
Wtrace_0
Xtrace_1
Ytrace_2
Ztrace_32э
&__inference_gru_layer_call_fn_19998716
&__inference_gru_layer_call_fn_19998727
&__inference_gru_layer_call_fn_19998738
&__inference_gru_layer_call_fn_19998749╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zWtrace_0zXtrace_1zYtrace_2zZtrace_3
─
[trace_0
\trace_1
]trace_2
^trace_32┘
A__inference_gru_layer_call_and_return_conditional_losses_19998902
A__inference_gru_layer_call_and_return_conditional_losses_19999055
A__inference_gru_layer_call_and_return_conditional_losses_19999208
A__inference_gru_layer_call_and_return_conditional_losses_19999361╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z[trace_0z\trace_1z]trace_2z^trace_3
C
#__self_saveable_object_factories"
_generic_user_object
Н
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
f_random_generator

=kernel
>recurrent_kernel
?bias
#g_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╗
mtrace_0
ntrace_12Д
*__inference_dropout_layer_call_fn_19999366
*__inference_dropout_layer_call_fn_19999371й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zmtrace_0zntrace_1
ё
otrace_0
ptrace_12║
E__inference_dropout_layer_call_and_return_conditional_losses_19999383
E__inference_dropout_layer_call_and_return_conditional_losses_19999388й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zotrace_0zptrace_1
C
#q_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
5
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

rstates
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
р
xtrace_0
ytrace_1
ztrace_2
{trace_32ї
(__inference_gru_1_layer_call_fn_19999399
(__inference_gru_1_layer_call_fn_19999410
(__inference_gru_1_layer_call_fn_19999421
(__inference_gru_1_layer_call_fn_19999432╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zxtrace_0zytrace_1zztrace_2z{trace_3
╠
|trace_0
}trace_1
~trace_2
trace_32с
C__inference_gru_1_layer_call_and_return_conditional_losses_19999587
C__inference_gru_1_layer_call_and_return_conditional_losses_19999742
C__inference_gru_1_layer_call_and_return_conditional_losses_19999897
C__inference_gru_1_layer_call_and_return_conditional_losses_20000052╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z|trace_0z}trace_1z~trace_2ztrace_3
D
$А_self_saveable_object_factories"
_generic_user_object
Х
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses
З_random_generator

@kernel
Arecurrent_kernel
Bbias
$И_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
├
Оtrace_0
Пtrace_12И
,__inference_dropout_1_layer_call_fn_20000057
,__inference_dropout_1_layer_call_fn_20000062й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zОtrace_0zПtrace_1
∙
Рtrace_0
Сtrace_12╛
G__inference_dropout_1_layer_call_and_return_conditional_losses_20000074
G__inference_dropout_1_layer_call_and_return_conditional_losses_20000079й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zРtrace_0zСtrace_1
D
$Т_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
ф
Шtrace_02┼
(__inference_dense_layer_call_fn_20000088Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zШtrace_0
 
Щtrace_02р
C__inference_dense_layer_call_and_return_conditional_losses_20000098Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЩtrace_0
:	╚2dense/kernel
:2
dense/bias
 "
trackable_dict_wrapper
&:$	╪2gru/gru_cell/kernel
1:/
╚╪2gru/gru_cell/recurrent_kernel
$:"	╪2gru/gru_cell/bias
+:)
╚╪2gru_1/gru_cell_1/kernel
5:3
╚╪2!gru_1/gru_cell_1/recurrent_kernel
(:&	╪2gru_1/gru_cell_1/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
(
Ъ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ўBЇ
-__inference_sequential_layer_call_fn_19997838	gru_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
-__inference_sequential_layer_call_fn_19997884	gru_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
-__inference_sequential_layer_call_fn_19998046inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
-__inference_sequential_layer_call_fn_19998067inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
H__inference_sequential_layer_call_and_return_conditional_losses_19997448	gru_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
H__inference_sequential_layer_call_and_return_conditional_losses_19997791	gru_input"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ПBМ
H__inference_sequential_layer_call_and_return_conditional_losses_19998393inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ПBМ
H__inference_sequential_layer_call_and_return_conditional_losses_19998705inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧B╠
&__inference_signature_wrapper_19998025	gru_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ДBБ
&__inference_gru_layer_call_fn_19998716inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
&__inference_gru_layer_call_fn_19998727inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
&__inference_gru_layer_call_fn_19998738inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
&__inference_gru_layer_call_fn_19998749inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
A__inference_gru_layer_call_and_return_conditional_losses_19998902inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
A__inference_gru_layer_call_and_return_conditional_losses_19999055inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЭBЪ
A__inference_gru_layer_call_and_return_conditional_losses_19999208inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЭBЪ
A__inference_gru_layer_call_and_return_conditional_losses_19999361inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
5
=0
>1
?2"
trackable_list_wrapper
5
=0
>1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
╧
аtrace_0
бtrace_12Ф
-__inference_gru_cell_4_layer_call_fn_20000112
-__inference_gru_cell_4_layer_call_fn_20000126│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zаtrace_0zбtrace_1
Е
вtrace_0
гtrace_12╩
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_20000165
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_20000204│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0zгtrace_1
D
$д_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
*__inference_dropout_layer_call_fn_19999366inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
хBт
*__inference_dropout_layer_call_fn_19999371inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
E__inference_dropout_layer_call_and_return_conditional_losses_19999383inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
E__inference_dropout_layer_call_and_return_conditional_losses_19999388inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЖBГ
(__inference_gru_1_layer_call_fn_19999399inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
(__inference_gru_1_layer_call_fn_19999410inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
(__inference_gru_1_layer_call_fn_19999421inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
(__inference_gru_1_layer_call_fn_19999432inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
бBЮ
C__inference_gru_1_layer_call_and_return_conditional_losses_19999587inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
бBЮ
C__inference_gru_1_layer_call_and_return_conditional_losses_19999742inputs_0"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
C__inference_gru_1_layer_call_and_return_conditional_losses_19999897inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
C__inference_gru_1_layer_call_and_return_conditional_losses_20000052inputs"╩
├▓┐
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsв

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
5
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
╧
кtrace_0
лtrace_12Ф
-__inference_gru_cell_5_layer_call_fn_20000218
-__inference_gru_cell_5_layer_call_fn_20000232│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zкtrace_0zлtrace_1
Е
мtrace_0
нtrace_12╩
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_20000271
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_20000310│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zмtrace_0zнtrace_1
D
$о_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
,__inference_dropout_1_layer_call_fn_20000057inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
чBф
,__inference_dropout_1_layer_call_fn_20000062inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
G__inference_dropout_1_layer_call_and_return_conditional_losses_20000074inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
G__inference_dropout_1_layer_call_and_return_conditional_losses_20000079inputs"й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╥B╧
(__inference_dense_layer_call_fn_20000088inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
C__inference_dense_layer_call_and_return_conditional_losses_20000098inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
п	variables
░	keras_api

▒total

▓count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
-__inference_gru_cell_4_layer_call_fn_20000112inputsstates_0"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
-__inference_gru_cell_4_layer_call_fn_20000126inputsstates_0"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_20000165inputsstates_0"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_20000204inputsstates_0"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№B∙
-__inference_gru_cell_5_layer_call_fn_20000218inputsstates_0"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
-__inference_gru_cell_5_layer_call_fn_20000232inputsstates_0"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_20000271inputsstates_0"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЧBФ
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_20000310inputsstates_0"│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_dict_wrapper
0
▒0
▓1"
trackable_list_wrapper
.
п	variables"
_generic_user_object
:  (2total
:  (2countШ
#__inference__wrapped_model_19996397q?=>B@A:;6в3
,в)
'К$
	gru_input         
к "-к*
(
denseК
dense         л
C__inference_dense_layer_call_and_return_conditional_losses_20000098d:;0в-
&в#
!К
inputs         ╚
к ",в)
"К
tensor_0         
Ъ Е
(__inference_dense_layer_call_fn_20000088Y:;0в-
&в#
!К
inputs         ╚
к "!К
unknown         ░
G__inference_dropout_1_layer_call_and_return_conditional_losses_20000074e4в1
*в'
!К
inputs         ╚
p
к "-в*
#К 
tensor_0         ╚
Ъ ░
G__inference_dropout_1_layer_call_and_return_conditional_losses_20000079e4в1
*в'
!К
inputs         ╚
p 
к "-в*
#К 
tensor_0         ╚
Ъ К
,__inference_dropout_1_layer_call_fn_20000057Z4в1
*в'
!К
inputs         ╚
p
к ""К
unknown         ╚К
,__inference_dropout_1_layer_call_fn_20000062Z4в1
*в'
!К
inputs         ╚
p 
к ""К
unknown         ╚╢
E__inference_dropout_layer_call_and_return_conditional_losses_19999383m8в5
.в+
%К"
inputs         ╚
p
к "1в.
'К$
tensor_0         ╚
Ъ ╢
E__inference_dropout_layer_call_and_return_conditional_losses_19999388m8в5
.в+
%К"
inputs         ╚
p 
к "1в.
'К$
tensor_0         ╚
Ъ Р
*__inference_dropout_layer_call_fn_19999366b8в5
.в+
%К"
inputs         ╚
p
к "&К#
unknown         ╚Р
*__inference_dropout_layer_call_fn_19999371b8в5
.в+
%К"
inputs         ╚
p 
к "&К#
unknown         ╚╬
C__inference_gru_1_layer_call_and_return_conditional_losses_19999587ЖB@APвM
FвC
5Ъ2
0К-
inputs_0                  ╚

 
p

 
к "-в*
#К 
tensor_0         ╚
Ъ ╬
C__inference_gru_1_layer_call_and_return_conditional_losses_19999742ЖB@APвM
FвC
5Ъ2
0К-
inputs_0                  ╚

 
p 

 
к "-в*
#К 
tensor_0         ╚
Ъ ╜
C__inference_gru_1_layer_call_and_return_conditional_losses_19999897vB@A@в=
6в3
%К"
inputs         ╚

 
p

 
к "-в*
#К 
tensor_0         ╚
Ъ ╜
C__inference_gru_1_layer_call_and_return_conditional_losses_20000052vB@A@в=
6в3
%К"
inputs         ╚

 
p 

 
к "-в*
#К 
tensor_0         ╚
Ъ з
(__inference_gru_1_layer_call_fn_19999399{B@APвM
FвC
5Ъ2
0К-
inputs_0                  ╚

 
p

 
к ""К
unknown         ╚з
(__inference_gru_1_layer_call_fn_19999410{B@APвM
FвC
5Ъ2
0К-
inputs_0                  ╚

 
p 

 
к ""К
unknown         ╚Ч
(__inference_gru_1_layer_call_fn_19999421kB@A@в=
6в3
%К"
inputs         ╚

 
p

 
к ""К
unknown         ╚Ч
(__inference_gru_1_layer_call_fn_19999432kB@A@в=
6в3
%К"
inputs         ╚

 
p 

 
к ""К
unknown         ╚Х
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_20000165╚?=>]вZ
SвP
 К
inputs         
(в%
#К 
states_0         ╚
p
к "bв_
XвU
%К"

tensor_0_0         ╚
,Ъ)
'К$
tensor_0_1_0         ╚
Ъ Х
H__inference_gru_cell_4_layer_call_and_return_conditional_losses_20000204╚?=>]вZ
SвP
 К
inputs         
(в%
#К 
states_0         ╚
p 
к "bв_
XвU
%К"

tensor_0_0         ╚
,Ъ)
'К$
tensor_0_1_0         ╚
Ъ ь
-__inference_gru_cell_4_layer_call_fn_20000112║?=>]вZ
SвP
 К
inputs         
(в%
#К 
states_0         ╚
p
к "TвQ
#К 
tensor_0         ╚
*Ъ'
%К"

tensor_1_0         ╚ь
-__inference_gru_cell_4_layer_call_fn_20000126║?=>]вZ
SвP
 К
inputs         
(в%
#К 
states_0         ╚
p 
к "TвQ
#К 
tensor_0         ╚
*Ъ'
%К"

tensor_1_0         ╚Ц
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_20000271╔B@A^в[
TвQ
!К
inputs         ╚
(в%
#К 
states_0         ╚
p
к "bв_
XвU
%К"

tensor_0_0         ╚
,Ъ)
'К$
tensor_0_1_0         ╚
Ъ Ц
H__inference_gru_cell_5_layer_call_and_return_conditional_losses_20000310╔B@A^в[
TвQ
!К
inputs         ╚
(в%
#К 
states_0         ╚
p 
к "bв_
XвU
%К"

tensor_0_0         ╚
,Ъ)
'К$
tensor_0_1_0         ╚
Ъ э
-__inference_gru_cell_5_layer_call_fn_20000218╗B@A^в[
TвQ
!К
inputs         ╚
(в%
#К 
states_0         ╚
p
к "TвQ
#К 
tensor_0         ╚
*Ъ'
%К"

tensor_1_0         ╚э
-__inference_gru_cell_5_layer_call_fn_20000232╗B@A^в[
TвQ
!К
inputs         ╚
(в%
#К 
states_0         ╚
p 
к "TвQ
#К 
tensor_0         ╚
*Ъ'
%К"

tensor_1_0         ╚╪
A__inference_gru_layer_call_and_return_conditional_losses_19998902Т?=>OвL
EвB
4Ъ1
/К,
inputs_0                  

 
p

 
к ":в7
0К-
tensor_0                  ╚
Ъ ╪
A__inference_gru_layer_call_and_return_conditional_losses_19999055Т?=>OвL
EвB
4Ъ1
/К,
inputs_0                  

 
p 

 
к ":в7
0К-
tensor_0                  ╚
Ъ ╛
A__inference_gru_layer_call_and_return_conditional_losses_19999208y?=>?в<
5в2
$К!
inputs         

 
p

 
к "1в.
'К$
tensor_0         ╚
Ъ ╛
A__inference_gru_layer_call_and_return_conditional_losses_19999361y?=>?в<
5в2
$К!
inputs         

 
p 

 
к "1в.
'К$
tensor_0         ╚
Ъ ▓
&__inference_gru_layer_call_fn_19998716З?=>OвL
EвB
4Ъ1
/К,
inputs_0                  

 
p

 
к "/К,
unknown                  ╚▓
&__inference_gru_layer_call_fn_19998727З?=>OвL
EвB
4Ъ1
/К,
inputs_0                  

 
p 

 
к "/К,
unknown                  ╚Ш
&__inference_gru_layer_call_fn_19998738n?=>?в<
5в2
$К!
inputs         

 
p

 
к "&К#
unknown         ╚Ш
&__inference_gru_layer_call_fn_19998749n?=>?в<
5в2
$К!
inputs         

 
p 

 
к "&К#
unknown         ╚─
H__inference_sequential_layer_call_and_return_conditional_losses_19997448x?=>B@A:;>в;
4в1
'К$
	gru_input         
p

 
к ",в)
"К
tensor_0         
Ъ ─
H__inference_sequential_layer_call_and_return_conditional_losses_19997791x?=>B@A:;>в;
4в1
'К$
	gru_input         
p 

 
к ",в)
"К
tensor_0         
Ъ ┴
H__inference_sequential_layer_call_and_return_conditional_losses_19998393u?=>B@A:;;в8
1в.
$К!
inputs         
p

 
к ",в)
"К
tensor_0         
Ъ ┴
H__inference_sequential_layer_call_and_return_conditional_losses_19998705u?=>B@A:;;в8
1в.
$К!
inputs         
p 

 
к ",в)
"К
tensor_0         
Ъ Ю
-__inference_sequential_layer_call_fn_19997838m?=>B@A:;>в;
4в1
'К$
	gru_input         
p

 
к "!К
unknown         Ю
-__inference_sequential_layer_call_fn_19997884m?=>B@A:;>в;
4в1
'К$
	gru_input         
p 

 
к "!К
unknown         Ы
-__inference_sequential_layer_call_fn_19998046j?=>B@A:;;в8
1в.
$К!
inputs         
p

 
к "!К
unknown         Ы
-__inference_sequential_layer_call_fn_19998067j?=>B@A:;;в8
1в.
$К!
inputs         
p 

 
к "!К
unknown         и
&__inference_signature_wrapper_19998025~?=>B@A:;Cв@
в 
9к6
4
	gru_input'К$
	gru_input         "-к*
(
denseК
dense         
ЇО
Ъ¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*1.15.02unknown8·╤
u
dense/kernelVarHandleOp*
shared_namedense/kernel*
_output_shapes
: *
shape:	А*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	А*
dtype0
m

dense/biasVarHandleOp*
dtype0*
shared_name
dense/bias*
_output_shapes
: *
shape:А
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
Л
layer_normalization/gammaVarHandleOp**
shared_namelayer_normalization/gamma*
dtype0*
shape:А*
_output_shapes
: 
Д
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes	
:А*
dtype0
Й
layer_normalization/betaVarHandleOp*
shape:А*)
shared_namelayer_normalization/beta*
dtype0*
_output_shapes
: 
В
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes	
:А*
dtype0
z
dense_1/kernelVarHandleOp*
dtype0*
shape:
АА*
shared_namedense_1/kernel*
_output_shapes
: 
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
АА*
dtype0
q
dense_1/biasVarHandleOp*
shape:А*
_output_shapes
: *
dtype0*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:А
П
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
shape:А*
dtype0*,
shared_namelayer_normalization_1/gamma
И
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
dtype0*
_output_shapes	
:А
Н
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namelayer_normalization_1/beta
Ж
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
dtype0*
_output_shapes	
:А
y
dense_2/kernelVarHandleOp*
shape:	А*
shared_namedense_2/kernel*
dtype0*
_output_shapes
: 
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	А*
dtype0
p
dense_2/biasVarHandleOp*
shared_namedense_2/bias*
shape:*
_output_shapes
: *
dtype0
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
dtype0*
shared_namedense_3/kernel*
_output_shapes
: *
shape:	А
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	А*
dtype0
p
dense_3/biasVarHandleOp*
shape:*
shared_namedense_3/bias*
dtype0*
_output_shapes
: 
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Гa
ConstConst"/device:CPU:0*
dtype0*╛`
value┤`B▒` Bк`
┤
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer_with_weights-5
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%
signatures
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
~

*kernel
+_callable_losses
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
З
1axis
	2gamma
3beta
4_callable_losses
5	variables
6trainable_variables
7regularization_losses
8	keras_api
h
9_callable_losses
:	variables
;trainable_variables
<regularization_losses
=	keras_api
~

>kernel
?_callable_losses
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
З
Eaxis
	Fgamma
Gbeta
H_callable_losses
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
h
M_callable_losses
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
~

Rkernel
Sbias
T_callable_losses
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
w
Y	constants
Z_callable_losses
[	variables
\trainable_variables
]regularization_losses
^	keras_api
w
_	constants
`_callable_losses
a	variables
btrainable_variables
cregularization_losses
d	keras_api
w
e	constants
f_callable_losses
g	variables
htrainable_variables
iregularization_losses
j	keras_api
w
k	constants
l_callable_losses
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
w
q	constants
r_callable_losses
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w
w	constants
x_callable_losses
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
z
}	constants
~_callable_losses
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
}
Г	constants
Д_callable_losses
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
}
Й	constants
К_callable_losses
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
}
П	constants
Р_callable_losses
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
}
Х	constants
Ц_callable_losses
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
}
Ы	constants
Ь_callable_losses
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
}
б	constants
в_callable_losses
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
Е
зkernel
	иbias
й_callable_losses
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
}
о	constants
п_callable_losses
░	variables
▒trainable_variables
▓regularization_losses
│	keras_api
}
┤	constants
╡_callable_losses
╢	variables
╖trainable_variables
╕regularization_losses
╣	keras_api
}
║	constants
╗_callable_losses
╝	variables
╜trainable_variables
╛regularization_losses
┐	keras_api
}
└	constants
┴_callable_losses
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
}
╞	constants
╟_callable_losses
╚	variables
╔trainable_variables
╩regularization_losses
╦	keras_api
}
╠	constants
═_callable_losses
╬	variables
╧trainable_variables
╨regularization_losses
╤	keras_api
}
╥	constants
╙_callable_losses
╘	variables
╒trainable_variables
╓regularization_losses
╫	keras_api
}
╪	constants
┘_callable_losses
┌	variables
█trainable_variables
▄regularization_losses
▌	keras_api
}
▐	constants
▀_callable_losses
р	variables
сtrainable_variables
тregularization_losses
у	keras_api
}
ф	constants
х_callable_losses
ц	variables
чtrainable_variables
шregularization_losses
щ	keras_api
X
*0
,1
22
33
>4
@5
F6
G7
R8
S9
з10
и11
X
*0
,1
22
33
>4
@5
F6
G7
R8
S9
з10
и11
 
Ю
!	variables
 ъlayer_regularization_losses
ыlayers
ьmetrics
"trainable_variables
#regularization_losses
эnon_trainable_variables
 
 
 
 
Ю
&	variables
 юlayer_regularization_losses
яlayers
Ёmetrics
'trainable_variables
(regularization_losses
ёnon_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
,1

*0
,1
 
Ю
-	variables
 Єlayer_regularization_losses
єlayers
Їmetrics
.trainable_variables
/regularization_losses
їnon_trainable_variables
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
 

20
31

20
31
 
Ю
5	variables
 Ўlayer_regularization_losses
ўlayers
°metrics
6trainable_variables
7regularization_losses
∙non_trainable_variables
 
 
 
 
Ю
:	variables
 ·layer_regularization_losses
√layers
№metrics
;trainable_variables
<regularization_losses
¤non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
@1

>0
@1
 
Ю
A	variables
 ■layer_regularization_losses
 layers
Аmetrics
Btrainable_variables
Cregularization_losses
Бnon_trainable_variables
 
fd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
 
Ю
I	variables
 Вlayer_regularization_losses
Гlayers
Дmetrics
Jtrainable_variables
Kregularization_losses
Еnon_trainable_variables
 
 
 
 
Ю
N	variables
 Жlayer_regularization_losses
Зlayers
Иmetrics
Otrainable_variables
Pregularization_losses
Йnon_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
 
Ю
U	variables
 Кlayer_regularization_losses
Лlayers
Мmetrics
Vtrainable_variables
Wregularization_losses
Нnon_trainable_variables
 
 
 
 
 
Ю
[	variables
 Оlayer_regularization_losses
Пlayers
Рmetrics
\trainable_variables
]regularization_losses
Сnon_trainable_variables
 
 
 
 
 
Ю
a	variables
 Тlayer_regularization_losses
Уlayers
Фmetrics
btrainable_variables
cregularization_losses
Хnon_trainable_variables
 
 
 
 
 
Ю
g	variables
 Цlayer_regularization_losses
Чlayers
Шmetrics
htrainable_variables
iregularization_losses
Щnon_trainable_variables
 
 
 
 
 
Ю
m	variables
 Ъlayer_regularization_losses
Ыlayers
Ьmetrics
ntrainable_variables
oregularization_losses
Эnon_trainable_variables
 
 
 
 
 
Ю
s	variables
 Юlayer_regularization_losses
Яlayers
аmetrics
ttrainable_variables
uregularization_losses
бnon_trainable_variables
 
 
 
 
 
Ю
y	variables
 вlayer_regularization_losses
гlayers
дmetrics
ztrainable_variables
{regularization_losses
еnon_trainable_variables
 
 
 
 
 
а
	variables
 жlayer_regularization_losses
зlayers
иmetrics
Аtrainable_variables
Бregularization_losses
йnon_trainable_variables
 
 
 
 
 
б
Е	variables
 кlayer_regularization_losses
лlayers
мmetrics
Жtrainable_variables
Зregularization_losses
нnon_trainable_variables
 
 
 
 
 
б
Л	variables
 оlayer_regularization_losses
пlayers
░metrics
Мtrainable_variables
Нregularization_losses
▒non_trainable_variables
 
 
 
 
 
б
С	variables
 ▓layer_regularization_losses
│layers
┤metrics
Тtrainable_variables
Уregularization_losses
╡non_trainable_variables
 
 
 
 
 
б
Ч	variables
 ╢layer_regularization_losses
╖layers
╕metrics
Шtrainable_variables
Щregularization_losses
╣non_trainable_variables
 
 
 
 
 
б
Э	variables
 ║layer_regularization_losses
╗layers
╝metrics
Юtrainable_variables
Яregularization_losses
╜non_trainable_variables
 
 
 
 
 
б
г	variables
 ╛layer_regularization_losses
┐layers
└metrics
дtrainable_variables
еregularization_losses
┴non_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

з0
и1

з0
и1
 
б
к	variables
 ┬layer_regularization_losses
├layers
─metrics
лtrainable_variables
мregularization_losses
┼non_trainable_variables
 
 
 
 
 
б
░	variables
 ╞layer_regularization_losses
╟layers
╚metrics
▒trainable_variables
▓regularization_losses
╔non_trainable_variables
 
 
 
 
 
б
╢	variables
 ╩layer_regularization_losses
╦layers
╠metrics
╖trainable_variables
╕regularization_losses
═non_trainable_variables
 
 
 
 
 
б
╝	variables
 ╬layer_regularization_losses
╧layers
╨metrics
╜trainable_variables
╛regularization_losses
╤non_trainable_variables
 
 
 
 
 
б
┬	variables
 ╥layer_regularization_losses
╙layers
╘metrics
├trainable_variables
─regularization_losses
╒non_trainable_variables
 
 
 
 
 
б
╚	variables
 ╓layer_regularization_losses
╫layers
╪metrics
╔trainable_variables
╩regularization_losses
┘non_trainable_variables
 
 
 
 
 
б
╬	variables
 ┌layer_regularization_losses
█layers
▄metrics
╧trainable_variables
╨regularization_losses
▌non_trainable_variables
 
 
 
 
 
б
╘	variables
 ▐layer_regularization_losses
▀layers
рmetrics
╒trainable_variables
╓regularization_losses
сnon_trainable_variables
 
 
 
 
 
б
┌	variables
 тlayer_regularization_losses
уlayers
фmetrics
█trainable_variables
▄regularization_losses
хnon_trainable_variables
 
 
 
 
 
б
р	variables
 цlayer_regularization_losses
чlayers
шmetrics
сtrainable_variables
тregularization_losses
щnon_trainable_variables
 
 
 
 
 
б
ц	variables
 ъlayer_regularization_losses
ыlayers
ьmetrics
чtrainable_variables
шregularization_losses
эnon_trainable_variables
 
Ў
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
_output_shapes
: 
Б
serving_default_policy_input_0Placeholder*
dtype0*
shape:         *'
_output_shapes
:         
С
StatefulPartitionedCallStatefulPartitionedCallserving_default_policy_input_0dense/kernel
dense/biaslayer_normalization/gammalayer_normalization/betadense_1/kerneldense_1/biaslayer_normalization_1/gammalayer_normalization_1/betadense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*,
f'R%
#__inference_signature_wrapper_72767*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-73643*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
shape: *
_output_shapes
: 
х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpConst*
Tout
2*'
f"R 
__inference__traced_save_73676*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-73677*
Tin
2
Ё
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biaslayer_normalization/gammalayer_normalization/betadense_1/kerneldense_1/biaslayer_normalization_1/gammalayer_normalization_1/betadense_2/kerneldense_2/biasdense_3/kerneldense_3/bias**
f%R#
!__inference__traced_restore_73725*
Tin
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-73726*
Tout
2*
_output_shapes
: Х╩
√
c
G__inference_activation_1_layer_call_and_return_conditional_losses_73255

inputs
identityG
ReluReluinputs*(
_output_shapes
:         А*
T0[
IdentityIdentityRelu:activations:0*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
▓
х
B__inference_dense_2_layer_call_and_return_conditional_losses_71868

inputs(
$matmul_readvariableop_dense_2_kernel'
#biasadd_readvariableop_dense_2_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
dtype0*
_output_shapes
:	Аi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Є
]
?__inference_tf_op_layer_Normal_1/sample/mul_layer_call_fn_73398
inputs_0
identity┤
PartitionedCallPartitionedCallinputs_0*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *
Tin
2*c
f^R\
Z__inference_tf_op_layer_Normal_1/sample/mul_layer_call_and_return_conditional_losses_72103*,
_gradient_op_typePartitionedCall-72110*
Tout
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*/
_input_shapes
:                  :( $
"
_user_specified_name
inputs/0
о
W
9__inference_tf_op_layer_strided_slice_layer_call_fn_73322
inputs_0
identityЭ
PartitionedCallPartitionedCallinputs_0*,
_gradient_op_typePartitionedCall-71964*
_output_shapes
: *]
fXRV
T__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_71957*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2O
IdentityIdentityPartitionedCall:output:0*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
::( $
"
_user_specified_name
inputs/0
Ж
r
T__inference_tf_op_layer_clip_by_value_layer_call_and_return_conditional_losses_73294
inputs_0
identityT
clip_by_value/yConst*
valueB
 *   └*
_output_shapes
: *
dtype0n
clip_by_valueMaximuminputs_0clip_by_value/y:output:0*'
_output_shapes
:         *
T0Y
IdentityIdentityclip_by_value:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
▓
Щ
^__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_fn_73365
inputs_0
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputs_0*-
config_proto

GPU

CPU2*0J 8*В
f}R{
y__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_and_return_conditional_losses_72040*
Tout
2*0
_output_shapes
:                  *,
_gradient_op_typePartitionedCall-72047*
Tin
2Л
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
╦
N
0__inference_tf_op_layer_Tanh_layer_call_fn_73551
inputs_0
identityе
PartitionedCallPartitionedCallinputs_0*
Tout
2*,
_gradient_op_typePartitionedCall-72395*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         *T
fORM
K__inference_tf_op_layer_Tanh_layer_call_and_return_conditional_losses_72388`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
НЮ
м

B__inference_model_1_layer_call_and_return_conditional_losses_72436
policy_input_0.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_biasI
Elayer_normalization_statefulpartitionedcall_layer_normalization_gammaH
Dlayer_normalization_statefulpartitionedcall_layer_normalization_beta2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_biasM
Ilayer_normalization_1_statefulpartitionedcall_layer_normalization_1_gammaL
Hlayer_normalization_1_statefulpartitionedcall_layer_normalization_1_beta2
.dense_2_statefulpartitionedcall_dense_2_kernel0
,dense_2_statefulpartitionedcall_dense_2_bias2
.dense_3_statefulpartitionedcall_dense_3_kernel0
,dense_3_statefulpartitionedcall_dense_3_bias
identityИвdense/StatefulPartitionedCallв,dense/bias/Regularizer/Square/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpвdense_1/StatefulPartitionedCallв.dense_1/bias/Regularizer/Square/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallвVtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallП
dense/StatefulPartitionedCallStatefulPartitionedCallpolicy_input_0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_71672*,
_gradient_op_typePartitionedCall-71679·
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0Elayer_normalization_statefulpartitionedcall_layer_normalization_gammaDlayer_normalization_statefulpartitionedcall_layer_normalization_beta*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         А*,
_gradient_op_typePartitionedCall-71720*
Tin
2*W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_71713╫
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tout
2*,
_gradient_op_typePartitionedCall-71744*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_71737*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*
Tin
2░
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_71778*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*(
_output_shapes
:         А*,
_gradient_op_typePartitionedCall-71785И
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0Ilayer_normalization_1_statefulpartitionedcall_layer_normalization_1_gammaHlayer_normalization_1_statefulpartitionedcall_layer_normalization_1_beta*
Tout
2*Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_71819*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-71826*
Tin
2▌
activation_1/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         А*,
_gradient_op_typePartitionedCall-71850*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_71843*
Tin
2*
Tout
2▒
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0.dense_2_statefulpartitionedcall_dense_2_kernel,dense_2_statefulpartitionedcall_dense_2_bias*
Tin
2*
Tout
2*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_71868*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-71875°
1tf_op_layer_clip_by_value/Minimum/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-71900*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*e
f`R^
\__inference_tf_op_layer_clip_by_value/Minimum_layer_call_and_return_conditional_losses_71893*
Tout
2*
Tin
2·
)tf_op_layer_clip_by_value/PartitionedCallPartitionedCall:tf_op_layer_clip_by_value/Minimum/PartitionedCall:output:0*]
fXRV
T__inference_tf_op_layer_clip_by_value_layer_call_and_return_conditional_losses_71914*,
_gradient_op_typePartitionedCall-71921*'
_output_shapes
:         *
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2╒
!tf_op_layer_Shape/PartitionedCallPartitionedCall2tf_op_layer_clip_by_value/PartitionedCall:output:0*
Tin
2*,
_gradient_op_typePartitionedCall-71941*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
:*U
fPRN
L__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_71934*
Tout
2┘
)tf_op_layer_strided_slice/PartitionedCallPartitionedCall*tf_op_layer_Shape/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-71964*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
_output_shapes
: *]
fXRV
T__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_71957я
0tf_op_layer_Normal_1/sample/Prod/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*d
f_R]
[__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_and_return_conditional_losses_71978*
Tout
2*,
_gradient_op_typePartitionedCall-71985*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
: *
Tin
2Р
;tf_op_layer_Normal_1/sample/concat/values_0/PartitionedCallPartitionedCall9tf_op_layer_Normal_1/sample/Prod/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72005*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*
_output_shapes
:*o
fjRh
f__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_and_return_conditional_losses_71998Й
2tf_op_layer_Normal_1/sample/concat/PartitionedCallPartitionedCallDtf_op_layer_Normal_1/sample/concat/values_0/PartitionedCall:output:0*f
faR_
]__inference_tf_op_layer_Normal_1/sample/concat_layer_call_and_return_conditional_losses_72020*
_output_shapes
:*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-72027▀
Vtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallStatefulPartitionedCall;tf_op_layer_Normal_1/sample/concat/PartitionedCall:output:0*0
_output_shapes
:                  *
Tin
2*,
_gradient_op_typePartitionedCall-72047*
Tout
2*-
config_proto

GPU

CPU2*0J 8*В
f}R{
y__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_and_return_conditional_losses_72040╨
=tf_op_layer_Normal_1/sample/random_normal/mul/PartitionedCallPartitionedCall_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tout
2*q
flRj
h__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_and_return_conditional_losses_72061*
Tin
2*,
_gradient_op_typePartitionedCall-72068*0
_output_shapes
:                  п
9tf_op_layer_Normal_1/sample/random_normal/PartitionedCallPartitionedCallFtf_op_layer_Normal_1/sample/random_normal/mul/PartitionedCall:output:0*m
fhRf
d__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_and_return_conditional_losses_72082*0
_output_shapes
:                  *
Tout
2*,
_gradient_op_typePartitionedCall-72089*
Tin
2*-
config_proto

GPU

CPU2*0J 8О
/tf_op_layer_Normal_1/sample/mul/PartitionedCallPartitionedCallBtf_op_layer_Normal_1/sample/random_normal/PartitionedCall:output:0*
Tin
2*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-72110*
Tout
2*c
f^R\
Z__inference_tf_op_layer_Normal_1/sample/mul_layer_call_and_return_conditional_losses_72103*-
config_proto

GPU

CPU2*0J 8Д
/tf_op_layer_Normal_1/sample/add/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/mul/PartitionedCall:output:0*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *c
f^R\
Z__inference_tf_op_layer_Normal_1/sample/add_layer_call_and_return_conditional_losses_72124*,
_gradient_op_typePartitionedCall-72131▒
dense_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0.dense_3_statefulpartitionedcall_dense_3_kernel,dense_3_statefulpartitionedcall_dense_3_bias*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_72149*,
_gradient_op_typePartitionedCall-72156 
3tf_op_layer_Normal_1/sample/Shape_2/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/add/PartitionedCall:output:0*
_output_shapes
:*,
_gradient_op_typePartitionedCall-72180*-
config_proto

GPU

CPU2*0J 8*g
fbR`
^__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_and_return_conditional_losses_72173*
Tin
2*
Tout
2№
3tf_op_layer_clip_by_value_1/Minimum/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*'
_output_shapes
:         *g
fbR`
^__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_and_return_conditional_losses_72194*
Tin
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72201*
Tout
2Ы
Dtf_op_layer_Normal_1/sample/expand_to_vector/Reshape/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
:*x
fsRq
o__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_and_return_conditional_losses_72215*,
_gradient_op_typePartitionedCall-72222*
Tout
2*
Tin
2П
9tf_op_layer_Normal_1/sample/strided_slice/PartitionedCallPartitionedCall<tf_op_layer_Normal_1/sample/Shape_2/PartitionedCall:output:0*
Tin
2*,
_gradient_op_typePartitionedCall-72245*m
fhRf
d__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_and_return_conditional_losses_72238*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
:А
+tf_op_layer_clip_by_value_1/PartitionedCallPartitionedCall<tf_op_layer_clip_by_value_1/Minimum/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72266*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *
Tout
2*
Tin
2*_
fZRX
V__inference_tf_op_layer_clip_by_value_1_layer_call_and_return_conditional_losses_72259█
4tf_op_layer_Normal_1/sample/concat_1/PartitionedCallPartitionedCallMtf_op_layer_Normal_1/sample/expand_to_vector/Reshape/PartitionedCall:output:0Btf_op_layer_Normal_1/sample/strided_slice/PartitionedCall:output:0*
_output_shapes
:*h
fcRa
___inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_and_return_conditional_losses_72281*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72289*
Tout
2*
Tin
2╒
3tf_op_layer_Normal_1/sample/Reshape/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/add/PartitionedCall:output:0=tf_op_layer_Normal_1/sample/concat_1/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2*g
fbR`
^__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_and_return_conditional_losses_72303*0
_output_shapes
:                  *,
_gradient_op_typePartitionedCall-72311р
tf_op_layer_Exp/PartitionedCallPartitionedCall4tf_op_layer_clip_by_value_1/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72331*S
fNRL
J__inference_tf_op_layer_Exp_layer_call_and_return_conditional_losses_72324*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*'
_output_shapes
:         У
tf_op_layer_mul/PartitionedCallPartitionedCall<tf_op_layer_Normal_1/sample/Reshape/PartitionedCall:output:0(tf_op_layer_Exp/PartitionedCall:output:0*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-72353*S
fNRL
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_72345*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8Й
tf_op_layer_add/PartitionedCallPartitionedCall2tf_op_layer_clip_by_value/PartitionedCall:output:0(tf_op_layer_mul/PartitionedCall:output:0*S
fNRL
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_72367*
Tout
2*
Tin
2*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72375╓
 tf_op_layer_Tanh/PartitionedCallPartitionedCall(tf_op_layer_add/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tout
2*,
_gradient_op_typePartitionedCall-72395*'
_output_shapes
:         *T
fORM
K__inference_tf_op_layer_Tanh_layer_call_and_return_conditional_losses_72388*
Tin
2║
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_statefulpartitionedcall_dense_kernel^dense/StatefulPartitionedCall*
dtype0*
_output_shapes
:	АЛ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes
:	А*
T0o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: С
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0▓
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_statefulpartitionedcall_dense_bias^dense/StatefulPartitionedCall*
_output_shapes	
:А*
dtype0Г
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
valueB: *
dtype0М
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
_output_shapes
: *
T0a
dense/bias/Regularizer/mul/xConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: О
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
_output_shapes
: *
T0a
dense/bias/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0Л
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0├
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.dense_1_statefulpartitionedcall_dense_1_kernel ^dense_1/StatefulPartitionedCall*
dtype0* 
_output_shapes
:
ААР
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0* 
_output_shapes
:
АА*
T0q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0Ш
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/mul/xConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: Ъ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: Ч
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0║
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp,dense_1_statefulpartitionedcall_dense_1_bias ^dense_1/StatefulPartitionedCall*
dtype0*
_output_shapes	
:АЗ
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аh
dense_1/bias/Regularizer/ConstConst*
valueB: *
dtype0*
_output_shapes
:Т
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: Ф
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: С
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0Є
IdentityIdentity)tf_op_layer_Tanh/PartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCallW^tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2░
Vtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallVtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall: :. *
(
_user_specified_namepolicy_input_0: : : : : : : : :	 :
 : 
ч
j
L__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_73304
inputs_0
identity=
ShapeShapeinputs_0*
_output_shapes
:*
T0I
IdentityIdentityShape:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
│
O
1__inference_tf_op_layer_Shape_layer_call_fn_73309
inputs_0
identityЩ
PartitionedCallPartitionedCallinputs_0*-
config_proto

GPU

CPU2*0J 8*
Tin
2*U
fPRN
L__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_71934*
Tout
2*,
_gradient_op_typePartitionedCall-71941*
_output_shapes
:S
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
╚
╔
B__inference_dense_1_layer_call_and_return_conditional_losses_71778

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв.dense_1/bias/Regularizer/Square/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel*
dtype0* 
_output_shapes
:
ААj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0w
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ап
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel^MatMul/ReadVariableOp* 
_output_shapes
:
АА*
dtype0Р
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0* 
_output_shapes
:
АА*
T0q
 dense_1/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       Ш
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *oГ:*
dtype0Ъ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: Ч
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0и
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias^BiasAdd/ReadVariableOp*
_output_shapes	
:А*
dtype0З
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0h
dense_1/bias/Regularizer/ConstConst*
dtype0*
valueB: *
_output_shapes
:Т
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Ф
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    С
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0ю
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ї
В
d__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_and_return_conditional_losses_73382
inputs_0
identityg
"Normal_1/sample/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    Ц
Normal_1/sample/random_normalAddinputs_0+Normal_1/sample/random_normal/mean:output:0*0
_output_shapes
:                  *
T0r
IdentityIdentity!Normal_1/sample/random_normal:z:0*0
_output_shapes
:                  *
T0"
identityIdentity:output:0*/
_input_shapes
:                  :( $
"
_user_specified_name
inputs/0
█
Ъ
N__inference_layer_normalization_layer_call_and_return_conditional_losses_71713

inputs:
6batchnorm_mul_readvariableop_layer_normalization_gamma5
1batchnorm_readvariableop_layer_normalization_beta
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0И
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
	keep_dims(*
T0*'
_output_shapes
:         m
moments/StopGradientStopGradientmoments/mean:output:0*'
_output_shapes
:         *
T0И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Аl
"moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:з
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*'
_output_shapes
:         *
	keep_dims(*
T0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:}
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*'
_output_shapes
:         *
T0]
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*'
_output_shapes
:         Р
batchnorm/mul/ReadVariableOpReadVariableOp6batchnorm_mul_readvariableop_layer_normalization_gamma*
dtype0*
_output_shapes	
:АВ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*(
_output_shapes
:         А*
T0s
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*(
_output_shapes
:         А*
T0З
batchnorm/ReadVariableOpReadVariableOp1batchnorm_readvariableop_layer_normalization_beta*
dtype0*
_output_shapes	
:А~
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         АЦ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
а
v
Z__inference_tf_op_layer_Normal_1/sample/add_layer_call_and_return_conditional_losses_72124

inputs
identityn
zerosConst*5
value,B*"                                 *
_output_shapes
:*
dtype0f
Normal_1/sample/addAddV2inputszeros:output:0*
T0*'
_output_shapes
:         _
IdentityIdentityNormal_1/sample/add:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
М
И
^__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_and_return_conditional_losses_72303

inputs
inputs_1
identityo
Normal_1/sample/ReshapeReshapeinputsinputs_1*
T0*0
_output_shapes
:                  q
IdentityIdentity Normal_1/sample/Reshape:output:0*0
_output_shapes
:                  *
T0"
identityIdentity:output:0*,
_input_shapes
:         ::& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
╫
a
C__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_fn_73419
inputs_0
identityл
PartitionedCallPartitionedCallinputs_0*
Tout
2*
Tin
2*
_output_shapes
:*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72180*g
fbR`
^__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_and_return_conditional_losses_72173S
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
їЭ
д

B__inference_model_1_layer_call_and_return_conditional_losses_72600

inputs.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_biasI
Elayer_normalization_statefulpartitionedcall_layer_normalization_gammaH
Dlayer_normalization_statefulpartitionedcall_layer_normalization_beta2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_biasM
Ilayer_normalization_1_statefulpartitionedcall_layer_normalization_1_gammaL
Hlayer_normalization_1_statefulpartitionedcall_layer_normalization_1_beta2
.dense_2_statefulpartitionedcall_dense_2_kernel0
,dense_2_statefulpartitionedcall_dense_2_bias2
.dense_3_statefulpartitionedcall_dense_3_kernel0
,dense_3_statefulpartitionedcall_dense_3_bias
identityИвdense/StatefulPartitionedCallв,dense/bias/Regularizer/Square/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpвdense_1/StatefulPartitionedCallв.dense_1/bias/Regularizer/Square/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallвVtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallЗ
dense/StatefulPartitionedCallStatefulPartitionedCallinputs*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*,
_gradient_op_typePartitionedCall-71679*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         А*
Tin
2*
Tout
2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_71672·
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0Elayer_normalization_statefulpartitionedcall_layer_normalization_gammaDlayer_normalization_statefulpartitionedcall_layer_normalization_beta*
Tout
2*,
_gradient_op_typePartitionedCall-71720*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*
Tin
2*W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_71713╫
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-71744*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_71737*
Tin
2*
Tout
2*(
_output_shapes
:         А░
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_71778*
Tout
2*
Tin
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-71785*(
_output_shapes
:         АИ
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0Ilayer_normalization_1_statefulpartitionedcall_layer_normalization_1_gammaHlayer_normalization_1_statefulpartitionedcall_layer_normalization_1_beta*
Tin
2*-
config_proto

GPU

CPU2*0J 8*Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_71819*(
_output_shapes
:         А*
Tout
2*,
_gradient_op_typePartitionedCall-71826▌
activation_1/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         А*
Tout
2*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_71843*
Tin
2*,
_gradient_op_typePartitionedCall-71850▒
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0.dense_2_statefulpartitionedcall_dense_2_kernel,dense_2_statefulpartitionedcall_dense_2_bias*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-71875*
Tout
2*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_71868*
Tin
2°
1tf_op_layer_clip_by_value/Minimum/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*e
f`R^
\__inference_tf_op_layer_clip_by_value/Minimum_layer_call_and_return_conditional_losses_71893*
Tout
2*
Tin
2*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-71900*-
config_proto

GPU

CPU2*0J 8·
)tf_op_layer_clip_by_value/PartitionedCallPartitionedCall:tf_op_layer_clip_by_value/Minimum/PartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-71921*]
fXRV
T__inference_tf_op_layer_clip_by_value_layer_call_and_return_conditional_losses_71914*'
_output_shapes
:         *
Tin
2╒
!tf_op_layer_Shape/PartitionedCallPartitionedCall2tf_op_layer_clip_by_value/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-71941*
_output_shapes
:*U
fPRN
L__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_71934┘
)tf_op_layer_strided_slice/PartitionedCallPartitionedCall*tf_op_layer_Shape/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-71964*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
: *]
fXRV
T__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_71957я
0tf_op_layer_Normal_1/sample/Prod/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
: *
Tin
2*d
f_R]
[__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_and_return_conditional_losses_71978*,
_gradient_op_typePartitionedCall-71985*
Tout
2Р
;tf_op_layer_Normal_1/sample/concat/values_0/PartitionedCallPartitionedCall9tf_op_layer_Normal_1/sample/Prod/PartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*o
fjRh
f__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_and_return_conditional_losses_71998*
Tin
2*,
_gradient_op_typePartitionedCall-72005*
_output_shapes
:Й
2tf_op_layer_Normal_1/sample/concat/PartitionedCallPartitionedCallDtf_op_layer_Normal_1/sample/concat/values_0/PartitionedCall:output:0*f
faR_
]__inference_tf_op_layer_Normal_1/sample/concat_layer_call_and_return_conditional_losses_72020*
Tout
2*,
_gradient_op_typePartitionedCall-72027*
_output_shapes
:*-
config_proto

GPU

CPU2*0J 8*
Tin
2▀
Vtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallStatefulPartitionedCall;tf_op_layer_Normal_1/sample/concat/PartitionedCall:output:0*
Tin
2*0
_output_shapes
:                  *-
config_proto

GPU

CPU2*0J 8*
Tout
2*,
_gradient_op_typePartitionedCall-72047*В
f}R{
y__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_and_return_conditional_losses_72040╨
=tf_op_layer_Normal_1/sample/random_normal/mul/PartitionedCallPartitionedCall_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall:output:0*
Tin
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72068*q
flRj
h__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_and_return_conditional_losses_72061*0
_output_shapes
:                  *
Tout
2п
9tf_op_layer_Normal_1/sample/random_normal/PartitionedCallPartitionedCallFtf_op_layer_Normal_1/sample/random_normal/mul/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72089*0
_output_shapes
:                  *m
fhRf
d__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_and_return_conditional_losses_72082*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8О
/tf_op_layer_Normal_1/sample/mul/PartitionedCallPartitionedCallBtf_op_layer_Normal_1/sample/random_normal/PartitionedCall:output:0*'
_output_shapes
:         *
Tout
2*
Tin
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72110*c
f^R\
Z__inference_tf_op_layer_Normal_1/sample/mul_layer_call_and_return_conditional_losses_72103Д
/tf_op_layer_Normal_1/sample/add/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/mul/PartitionedCall:output:0*
Tout
2*c
f^R\
Z__inference_tf_op_layer_Normal_1/sample/add_layer_call_and_return_conditional_losses_72124*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72131*
Tin
2*'
_output_shapes
:         ▒
dense_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0.dense_3_statefulpartitionedcall_dense_3_kernel,dense_3_statefulpartitionedcall_dense_3_bias*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_72149*,
_gradient_op_typePartitionedCall-72156*
Tin
2*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*
Tout
2 
3tf_op_layer_Normal_1/sample/Shape_2/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/add/PartitionedCall:output:0*
Tin
2*-
config_proto

GPU

CPU2*0J 8*g
fbR`
^__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_and_return_conditional_losses_72173*
Tout
2*
_output_shapes
:*,
_gradient_op_typePartitionedCall-72180№
3tf_op_layer_clip_by_value_1/Minimum/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*g
fbR`
^__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_and_return_conditional_losses_72194*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72201*
Tout
2*
Tin
2*'
_output_shapes
:         Ы
Dtf_op_layer_Normal_1/sample/expand_to_vector/Reshape/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72222*x
fsRq
o__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_and_return_conditional_losses_72215*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
:П
9tf_op_layer_Normal_1/sample/strided_slice/PartitionedCallPartitionedCall<tf_op_layer_Normal_1/sample/Shape_2/PartitionedCall:output:0*
_output_shapes
:*
Tin
2*-
config_proto

GPU

CPU2*0J 8*m
fhRf
d__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_and_return_conditional_losses_72238*
Tout
2*,
_gradient_op_typePartitionedCall-72245А
+tf_op_layer_clip_by_value_1/PartitionedCallPartitionedCall<tf_op_layer_clip_by_value_1/Minimum/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72266*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_tf_op_layer_clip_by_value_1_layer_call_and_return_conditional_losses_72259*
Tout
2*
Tin
2*'
_output_shapes
:         █
4tf_op_layer_Normal_1/sample/concat_1/PartitionedCallPartitionedCallMtf_op_layer_Normal_1/sample/expand_to_vector/Reshape/PartitionedCall:output:0Btf_op_layer_Normal_1/sample/strided_slice/PartitionedCall:output:0*
Tin
2*h
fcRa
___inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_and_return_conditional_losses_72281*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
_output_shapes
:*,
_gradient_op_typePartitionedCall-72289╒
3tf_op_layer_Normal_1/sample/Reshape/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/add/PartitionedCall:output:0=tf_op_layer_Normal_1/sample/concat_1/PartitionedCall:output:0*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*g
fbR`
^__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_and_return_conditional_losses_72303*0
_output_shapes
:                  *,
_gradient_op_typePartitionedCall-72311р
tf_op_layer_Exp/PartitionedCallPartitionedCall4tf_op_layer_clip_by_value_1/PartitionedCall:output:0*
Tin
2*,
_gradient_op_typePartitionedCall-72331*S
fNRL
J__inference_tf_op_layer_Exp_layer_call_and_return_conditional_losses_72324*'
_output_shapes
:         *
Tout
2*-
config_proto

GPU

CPU2*0J 8У
tf_op_layer_mul/PartitionedCallPartitionedCall<tf_op_layer_Normal_1/sample/Reshape/PartitionedCall:output:0(tf_op_layer_Exp/PartitionedCall:output:0*S
fNRL
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_72345*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-72353Й
tf_op_layer_add/PartitionedCallPartitionedCall2tf_op_layer_clip_by_value/PartitionedCall:output:0(tf_op_layer_mul/PartitionedCall:output:0*
Tout
2*
Tin
2*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-72375*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_72367╓
 tf_op_layer_Tanh/PartitionedCallPartitionedCall(tf_op_layer_add/PartitionedCall:output:0*
Tin
2*,
_gradient_op_typePartitionedCall-72395*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *
Tout
2*T
fORM
K__inference_tf_op_layer_Tanh_layer_call_and_return_conditional_losses_72388║
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_statefulpartitionedcall_dense_kernel^dense/StatefulPartitionedCall*
dtype0*
_output_shapes
:	АЛ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Аo
dense/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *oГ:*
dtype0Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: С
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ▓
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_statefulpartitionedcall_dense_bias^dense/StatefulPartitionedCall*
_output_shapes	
:А*
dtype0Г
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аf
dense/bias/Regularizer/ConstConst*
dtype0*
valueB: *
_output_shapes
:М
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
_output_shapes
: *
T0a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5О
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Л
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ├
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.dense_1_statefulpartitionedcall_dense_1_kernel ^dense_1/StatefulPartitionedCall*
dtype0* 
_output_shapes
:
ААР
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААq
 dense_1/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Ш
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
valueB
 *oГ:*
_output_shapes
: *
dtype0Ъ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: Ч
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ║
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp,dense_1_statefulpartitionedcall_dense_1_bias ^dense_1/StatefulPartitionedCall*
dtype0*
_output_shapes	
:АЗ
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0h
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
valueB: *
dtype0Т
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/mul/xConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: Ф
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0С
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0Є
IdentityIdentity)tf_op_layer_Tanh/PartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCallW^tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2░
Vtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallVtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:
 : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 
╚
`
B__inference_tf_op_layer_Normal_1/sample/concat_layer_call_fn_73355
inputs_0
identityк
PartitionedCallPartitionedCallinputs_0*
Tin
2*,
_gradient_op_typePartitionedCall-72027*-
config_proto

GPU

CPU2*0J 8*
Tout
2*f
faR_
]__inference_tf_op_layer_Normal_1/sample/concat_layer_call_and_return_conditional_losses_72020*
_output_shapes
:S
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*
_input_shapes
::( $
"
_user_specified_name
inputs/0
Я
[
/__inference_tf_op_layer_add_layer_call_fn_73541
inputs_0
inputs_1
identityп
PartitionedCallPartitionedCallinputs_0inputs_1*
Tout
2*,
_gradient_op_typePartitionedCall-72375*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_72367*
Tin
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*9
_input_shapes(
&:         :         :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
Ё
Т
'__inference_model_1_layer_call_fn_73067

inputs(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias5
1statefulpartitionedcall_layer_normalization_gamma4
0statefulpartitionedcall_layer_normalization_beta*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias7
3statefulpartitionedcall_layer_normalization_1_gamma6
2statefulpartitionedcall_layer_normalization_1_beta*
&statefulpartitionedcall_dense_2_kernel(
$statefulpartitionedcall_dense_2_bias*
&statefulpartitionedcall_dense_3_kernel(
$statefulpartitionedcall_dense_3_bias
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputs$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias1statefulpartitionedcall_layer_normalization_gamma0statefulpartitionedcall_layer_normalization_beta&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias3statefulpartitionedcall_layer_normalization_1_gamma2statefulpartitionedcall_layer_normalization_1_beta&statefulpartitionedcall_dense_2_kernel$statefulpartitionedcall_dense_2_bias&statefulpartitionedcall_dense_3_kernel$statefulpartitionedcall_dense_3_bias*,
_gradient_op_typePartitionedCall-72601*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*
Tin
2*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_72600*
Tout
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
э
_
A__inference_tf_op_layer_clip_by_value/Minimum_layer_call_fn_73288
inputs_0
identity╢
PartitionedCallPartitionedCallinputs_0*
Tout
2*e
f`R^
\__inference_tf_op_layer_clip_by_value/Minimum_layer_call_and_return_conditional_losses_71893*,
_gradient_op_typePartitionedCall-71900*
Tin
2*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
ъ
┐
@__inference_dense_layer_call_and_return_conditional_losses_73126

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв,dense/bias/Regularizer/Square/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0u
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0к
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel^MatMul/ReadVariableOp*
dtype0*
_output_shapes
:	АЛ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Аo
dense/kernel/Regularizer/ConstConst*
valueB"       *
_output_shapes
:*
dtype0Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/mul/xConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: С
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0д
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias^BiasAdd/ReadVariableOp*
_output_shapes	
:А*
dtype0Г
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0f
dense/bias/Regularizer/ConstConst*
valueB: *
dtype0*
_output_shapes
:М
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: О
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: Л
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0ъ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
╦
v
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_73535
inputs_0
inputs_1
identityR
addAddV2inputs_0inputs_1*'
_output_shapes
:         *
T0O
IdentityIdentityadd:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*9
_input_shapes(
&:         :         :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
Ё
Л
o__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_and_return_conditional_losses_72215

inputs
identityx
.Normal_1/sample/expand_to_vector/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0Щ
(Normal_1/sample/expand_to_vector/ReshapeReshapeinputs7Normal_1/sample/expand_to_vector/Reshape/shape:output:0*
_output_shapes
:*
T0l
IdentityIdentity1Normal_1/sample/expand_to_vector/Reshape:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*
_input_shapes
: :& "
 
_user_specified_nameinputs
╝
F
*__inference_activation_layer_call_fn_73172

inputs
identityЮ
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*(
_output_shapes
:         А*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_71737*,
_gradient_op_typePartitionedCall-71744a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
ю
w
[__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_and_return_conditional_losses_71978

inputs
identityX
Normal_1/sample/ConstConst*
dtype0*
valueB *
_output_shapes
: e
Normal_1/sample/ProdProdinputsNormal_1/sample/Const:output:0*
_output_shapes
: *
T0T
IdentityIdentityNormal_1/sample/Prod:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
: :& "
 
_user_specified_nameinputs
Є
g
K__inference_tf_op_layer_Tanh_layer_call_and_return_conditional_losses_72388

inputs
identityF
TanhTanhinputs*
T0*'
_output_shapes
:         P
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
и
x
\__inference_tf_op_layer_clip_by_value/Minimum_layer_call_and_return_conditional_losses_71893

inputs
identity\
clip_by_value/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @|
clip_by_value/MinimumMinimuminputs clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityclip_by_value/Minimum:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
є
╢
'__inference_dense_2_layer_call_fn_73277

inputs*
&statefulpartitionedcall_dense_2_kernel(
$statefulpartitionedcall_dense_2_bias
identityИвStatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputs&statefulpartitionedcall_dense_2_kernel$statefulpartitionedcall_dense_2_bias*,
_gradient_op_typePartitionedCall-71875*
Tin
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_71868*
Tout
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Р
t
V__inference_tf_op_layer_clip_by_value_1_layer_call_and_return_conditional_losses_73490
inputs_0
identityV
clip_by_value_1/yConst*
dtype0*
valueB
 *  а┴*
_output_shapes
: r
clip_by_value_1Maximuminputs_0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         [
IdentityIdentityclip_by_value_1:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
╞
Ъ
__inference_loss_fn_1_73580;
7dense_bias_regularizer_square_readvariableop_dense_bias
identityИв,dense/bias/Regularizer/Square/ReadVariableOpб
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp7dense_bias_regularizer_square_readvariableop_dense_bias*
dtype0*
_output_shapes	
:АГ
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
valueB: *
dtype0М
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: О
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
_output_shapes
: *
T0a
dense/bias/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0Л
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0Д
IdentityIdentitydense/bias/Regularizer/add:z:0-^dense/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp:  
з
В
f__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_and_return_conditional_losses_71998

inputs
identity]
Normal_1/sample/concat/values_0Packinputs*
_output_shapes
:*
N*
T0c
IdentityIdentity(Normal_1/sample/concat/values_0:output:0*
_output_shapes
:*
T0"
identityIdentity:output:0*
_input_shapes
: :& "
 
_user_specified_nameinputs
И
Ъ
'__inference_model_1_layer_call_fn_72616
policy_input_0(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias5
1statefulpartitionedcall_layer_normalization_gamma4
0statefulpartitionedcall_layer_normalization_beta*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias7
3statefulpartitionedcall_layer_normalization_1_gamma6
2statefulpartitionedcall_layer_normalization_1_beta*
&statefulpartitionedcall_dense_2_kernel(
$statefulpartitionedcall_dense_2_bias*
&statefulpartitionedcall_dense_3_kernel(
$statefulpartitionedcall_dense_3_bias
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallpolicy_input_0$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias1statefulpartitionedcall_layer_normalization_gamma0statefulpartitionedcall_layer_normalization_beta&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias3statefulpartitionedcall_layer_normalization_1_gamma2statefulpartitionedcall_layer_normalization_1_beta&statefulpartitionedcall_dense_2_kernel$statefulpartitionedcall_dense_2_bias&statefulpartitionedcall_dense_3_kernel$statefulpartitionedcall_dense_3_bias*,
_gradient_op_typePartitionedCall-72601*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*
Tout
2*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_72600*
Tin
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namepolicy_input_0: : : : : : : : :	 :
 : : 
й
Ш
y__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_and_return_conditional_losses_72040

inputs
identityИЪ
2Normal_1/sample/random_normal/RandomStandardNormalRandomStandardNormalinputs*
T0*
dtype0*0
_output_shapes
:                  М
IdentityIdentity;Normal_1/sample/random_normal/RandomStandardNormal:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*
_input_shapes
::& "
 
_user_specified_nameinputs
ж
x
Z__inference_tf_op_layer_Normal_1/sample/add_layer_call_and_return_conditional_losses_73404
inputs_0
identityn
zerosConst*
dtype0*
_output_shapes
:*5
value,B*"                                 h
Normal_1/sample/addAddV2inputs_0zeros:output:0*
T0*'
_output_shapes
:         _
IdentityIdentityNormal_1/sample/add:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
▓
х
B__inference_dense_3_layer_call_and_return_conditional_losses_73429

inputs(
$matmul_readvariableop_dense_3_kernel'
#biasadd_readvariableop_dense_3_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_3_kernel*
dtype0*
_output_shapes
:	Аi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_3_bias*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*/
_input_shapes
:         А::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ъ
┐
@__inference_dense_layer_call_and_return_conditional_losses_71672

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв,dense/bias/Regularizer/Square/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpy
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes
:	Аj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0u
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ак
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel^MatMul/ReadVariableOp*
dtype0*
_output_shapes
:	АЛ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes
:	А*
T0o
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
valueB"       *
dtype0Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0С
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0д
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias^BiasAdd/ReadVariableOp*
dtype0*
_output_shapes	
:АГ
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
valueB: *
dtype0М
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
valueB
 *╜7Ж5*
_output_shapes
: *
dtype0О
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0Л
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0ъ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
с
h
L__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_71934

inputs
identity;
ShapeShapeinputs*
_output_shapes
:*
T0I
IdentityIdentityShape:output:0*
_output_shapes
:*
T0"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
о
z
\__inference_tf_op_layer_clip_by_value/Minimum_layer_call_and_return_conditional_losses_73283
inputs_0
identity\
clip_by_value/Minimum/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0~
clip_by_value/MinimumMinimuminputs_0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         a
IdentityIdentityclip_by_value/Minimum:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
л
x
Z__inference_tf_op_layer_Normal_1/sample/mul_layer_call_and_return_conditional_losses_73393
inputs_0
identitym
onesConst*
_output_shapes
:*5
value,B*"   А?  А?  А?  А?  А?  А?  А?  А?*
dtype0e
Normal_1/sample/mulMulinputs_0ones:output:0*'
_output_shapes
:         *
T0_
IdentityIdentityNormal_1/sample/mul:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*/
_input_shapes
:                  :( $
"
_user_specified_name
inputs/0
ш
░
%__inference_dense_layer_call_fn_73133

inputs(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_71672*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-71679*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8Г
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
█
Ъ
N__inference_layer_normalization_layer_call_and_return_conditional_losses_73155

inputs:
6batchnorm_mul_readvariableop_layer_normalization_gamma5
1batchnorm_readvariableop_layer_normalization_beta
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0И
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
	keep_dims(*'
_output_shapes
:         m
moments/StopGradientStopGradientmoments/mean:output:0*'
_output_shapes
:         *
T0И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*(
_output_shapes
:         А*
T0l
"moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:з
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(T
batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: }
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*'
_output_shapes
:         *
T0]
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*'
_output_shapes
:         *
T0Р
batchnorm/mul/ReadVariableOpReadVariableOp6batchnorm_mul_readvariableop_layer_normalization_gamma*
_output_shapes	
:А*
dtype0В
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*(
_output_shapes
:         А*
T0s
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*(
_output_shapes
:         А*
T0З
batchnorm/ReadVariableOpReadVariableOp1batchnorm_readvariableop_layer_normalization_beta*
dtype0*
_output_shapes	
:А~
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*(
_output_shapes
:         А*
T0s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         АЦ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Їж
С
 __inference__wrapped_model_71639
policy_input_04
0model_1_dense_matmul_readvariableop_dense_kernel3
/model_1_dense_biasadd_readvariableop_dense_biasV
Rmodel_1_layer_normalization_batchnorm_mul_readvariableop_layer_normalization_gammaQ
Mmodel_1_layer_normalization_batchnorm_readvariableop_layer_normalization_beta8
4model_1_dense_1_matmul_readvariableop_dense_1_kernel7
3model_1_dense_1_biasadd_readvariableop_dense_1_biasZ
Vmodel_1_layer_normalization_1_batchnorm_mul_readvariableop_layer_normalization_1_gammaU
Qmodel_1_layer_normalization_1_batchnorm_readvariableop_layer_normalization_1_beta8
4model_1_dense_2_matmul_readvariableop_dense_2_kernel7
3model_1_dense_2_biasadd_readvariableop_dense_2_bias8
4model_1_dense_3_matmul_readvariableop_dense_3_kernel7
3model_1_dense_3_biasadd_readvariableop_dense_3_bias
identityИв$model_1/dense/BiasAdd/ReadVariableOpв#model_1/dense/MatMul/ReadVariableOpв&model_1/dense_1/BiasAdd/ReadVariableOpв%model_1/dense_1/MatMul/ReadVariableOpв&model_1/dense_2/BiasAdd/ReadVariableOpв%model_1/dense_2/MatMul/ReadVariableOpв&model_1/dense_3/BiasAdd/ReadVariableOpв%model_1/dense_3/MatMul/ReadVariableOpв4model_1/layer_normalization/batchnorm/ReadVariableOpв8model_1/layer_normalization/batchnorm/mul/ReadVariableOpв6model_1/layer_normalization_1/batchnorm/ReadVariableOpв:model_1/layer_normalization_1/batchnorm/mul/ReadVariableOpХ
#model_1/dense/MatMul/ReadVariableOpReadVariableOp0model_1_dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	А*
dtype0О
model_1/dense/MatMulMatMulpolicy_input_0+model_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АС
$model_1/dense/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:А*
dtype0б
model_1/dense/BiasAddBiasAddmodel_1/dense/MatMul:product:0,model_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АД
:model_1/layer_normalization/moments/mean/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0╪
(model_1/layer_normalization/moments/meanMeanmodel_1/dense/BiasAdd:output:0Cmodel_1/layer_normalization/moments/mean/reduction_indices:output:0*
T0*
	keep_dims(*'
_output_shapes
:         е
0model_1/layer_normalization/moments/StopGradientStopGradient1model_1/layer_normalization/moments/mean:output:0*'
_output_shapes
:         *
T0╪
5model_1/layer_normalization/moments/SquaredDifferenceSquaredDifferencemodel_1/dense/BiasAdd:output:09model_1/layer_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:         АИ
>model_1/layer_normalization/moments/variance/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0√
,model_1/layer_normalization/moments/varianceMean9model_1/layer_normalization/moments/SquaredDifference:z:0Gmodel_1/layer_normalization/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(p
+model_1/layer_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *oГ:╤
)model_1/layer_normalization/batchnorm/addAddV25model_1/layer_normalization/moments/variance:output:04model_1/layer_normalization/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         Х
+model_1/layer_normalization/batchnorm/RsqrtRsqrt-model_1/layer_normalization/batchnorm/add:z:0*
T0*'
_output_shapes
:         ╚
8model_1/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpRmodel_1_layer_normalization_batchnorm_mul_readvariableop_layer_normalization_gamma*
_output_shapes	
:А*
dtype0╓
)model_1/layer_normalization/batchnorm/mulMul/model_1/layer_normalization/batchnorm/Rsqrt:y:0@model_1/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А┤
+model_1/layer_normalization/batchnorm/mul_1Mulmodel_1/dense/BiasAdd:output:0-model_1/layer_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А╟
+model_1/layer_normalization/batchnorm/mul_2Mul1model_1/layer_normalization/moments/mean:output:0-model_1/layer_normalization/batchnorm/mul:z:0*(
_output_shapes
:         А*
T0┐
4model_1/layer_normalization/batchnorm/ReadVariableOpReadVariableOpMmodel_1_layer_normalization_batchnorm_readvariableop_layer_normalization_beta*
_output_shapes	
:А*
dtype0╥
)model_1/layer_normalization/batchnorm/subSub<model_1/layer_normalization/batchnorm/ReadVariableOp:value:0/model_1/layer_normalization/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         А╟
+model_1/layer_normalization/batchnorm/add_1AddV2/model_1/layer_normalization/batchnorm/mul_1:z:0-model_1/layer_normalization/batchnorm/sub:z:0*(
_output_shapes
:         А*
T0Г
model_1/activation/ReluRelu/model_1/layer_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         А*
T0Ь
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp4model_1_dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
АА*
dtype0й
model_1/dense_1/MatMulMatMul%model_1/activation/Relu:activations:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp3model_1_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:А*
dtype0з
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0Ж
<model_1/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0▐
*model_1/layer_normalization_1/moments/meanMean model_1/dense_1/BiasAdd:output:0Emodel_1/layer_normalization_1/moments/mean/reduction_indices:output:0*'
_output_shapes
:         *
	keep_dims(*
T0й
2model_1/layer_normalization_1/moments/StopGradientStopGradient3model_1/layer_normalization_1/moments/mean:output:0*
T0*'
_output_shapes
:         ▐
7model_1/layer_normalization_1/moments/SquaredDifferenceSquaredDifference model_1/dense_1/BiasAdd:output:0;model_1/layer_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:         АК
@model_1/layer_normalization_1/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:Б
.model_1/layer_normalization_1/moments/varianceMean;model_1/layer_normalization_1/moments/SquaredDifference:z:0Imodel_1/layer_normalization_1/moments/variance/reduction_indices:output:0*'
_output_shapes
:         *
T0*
	keep_dims(r
-model_1/layer_normalization_1/batchnorm/add/yConst*
dtype0*
valueB
 *oГ:*
_output_shapes
: ╫
+model_1/layer_normalization_1/batchnorm/addAddV27model_1/layer_normalization_1/moments/variance:output:06model_1/layer_normalization_1/batchnorm/add/y:output:0*'
_output_shapes
:         *
T0Щ
-model_1/layer_normalization_1/batchnorm/RsqrtRsqrt/model_1/layer_normalization_1/batchnorm/add:z:0*'
_output_shapes
:         *
T0╬
:model_1/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpVmodel_1_layer_normalization_1_batchnorm_mul_readvariableop_layer_normalization_1_gamma*
dtype0*
_output_shapes	
:А▄
+model_1/layer_normalization_1/batchnorm/mulMul1model_1/layer_normalization_1/batchnorm/Rsqrt:y:0Bmodel_1/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0║
-model_1/layer_normalization_1/batchnorm/mul_1Mul model_1/dense_1/BiasAdd:output:0/model_1/layer_normalization_1/batchnorm/mul:z:0*(
_output_shapes
:         А*
T0═
-model_1/layer_normalization_1/batchnorm/mul_2Mul3model_1/layer_normalization_1/moments/mean:output:0/model_1/layer_normalization_1/batchnorm/mul:z:0*(
_output_shapes
:         А*
T0┼
6model_1/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpQmodel_1_layer_normalization_1_batchnorm_readvariableop_layer_normalization_1_beta*
_output_shapes	
:А*
dtype0╪
+model_1/layer_normalization_1/batchnorm/subSub>model_1/layer_normalization_1/batchnorm/ReadVariableOp:value:01model_1/layer_normalization_1/batchnorm/mul_2:z:0*(
_output_shapes
:         А*
T0═
-model_1/layer_normalization_1/batchnorm/add_1AddV21model_1/layer_normalization_1/batchnorm/mul_1:z:0/model_1/layer_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         АЗ
model_1/activation_1/ReluRelu1model_1/layer_normalization_1/batchnorm/add_1:z:0*(
_output_shapes
:         А*
T0Ы
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp4model_1_dense_2_matmul_readvariableop_dense_2_kernel*
dtype0*
_output_shapes
:	Ак
model_1/dense_2/MatMulMatMul'model_1/activation_1/Relu:activations:0-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ц
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp3model_1_dense_2_biasadd_readvariableop_dense_2_bias*
dtype0*
_output_shapes
:ж
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Ж
Amodel_1/tf_op_layer_clip_by_value/Minimum/clip_by_value/Minimum/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: ъ
?model_1/tf_op_layer_clip_by_value/Minimum/clip_by_value/MinimumMinimum model_1/dense_2/BiasAdd:output:0Jmodel_1/tf_op_layer_clip_by_value/Minimum/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         v
1model_1/tf_op_layer_clip_by_value/clip_by_value/yConst*
valueB
 *   └*
dtype0*
_output_shapes
: э
/model_1/tf_op_layer_clip_by_value/clip_by_valueMaximumCmodel_1/tf_op_layer_clip_by_value/Minimum/clip_by_value/Minimum:z:0:model_1/tf_op_layer_clip_by_value/clip_by_value/y:output:0*'
_output_shapes
:         *
T0В
model_1/tf_op_layer_Shape/ShapeShape3model_1/tf_op_layer_clip_by_value/clip_by_value:z:0*
T0*
_output_shapes
:
5model_1/tf_op_layer_strided_slice/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: Б
7model_1/tf_op_layer_strided_slice/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:Б
7model_1/tf_op_layer_strided_slice/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
/model_1/tf_op_layer_strided_slice/strided_sliceStridedSlice(model_1/tf_op_layer_Shape/Shape:output:0>model_1/tf_op_layer_strided_slice/strided_slice/stack:output:0@model_1/tf_op_layer_strided_slice/strided_slice/stack_1:output:0@model_1/tf_op_layer_strided_slice/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: Б
>model_1/tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB щ
=model_1/tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/ProdProd8model_1/tf_op_layer_strided_slice/strided_slice:output:0Gmodel_1/tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/Const:output:0*
T0*
_output_shapes
: ╤
Smodel_1/tf_op_layer_Normal_1/sample/concat/values_0/Normal_1/sample/concat/values_0PackFmodel_1/tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/Prod:output:0*
N*
_output_shapes
:*
T0Т
Hmodel_1/tf_op_layer_Normal_1/sample/concat/Normal_1/sample/BroadcastArgsConst*
valueB:*
_output_shapes
:*
dtype0И
Fmodel_1/tf_op_layer_Normal_1/sample/concat/Normal_1/sample/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: ¤
Amodel_1/tf_op_layer_Normal_1/sample/concat/Normal_1/sample/concatConcatV2\model_1/tf_op_layer_Normal_1/sample/concat/values_0/Normal_1/sample/concat/values_0:output:0Qmodel_1/tf_op_layer_Normal_1/sample/concat/Normal_1/sample/BroadcastArgs:output:0Omodel_1/tf_op_layer_Normal_1/sample/concat/Normal_1/sample/concat/axis:output:0*
T0*
N*
_output_shapes
:Ь
ymodel_1/tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/Normal_1/sample/random_normal/RandomStandardNormalRandomStandardNormalJmodel_1/tf_op_layer_Normal_1/sample/concat/Normal_1/sample/concat:output:0*'
_output_shapes
:         *
T0*
dtype0Я
Zmodel_1/tf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: ·
Wmodel_1/tf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/mulMulВmodel_1/tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/Normal_1/sample/random_normal/RandomStandardNormal:output:0cmodel_1/tf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/stddev:output:0*
T0*'
_output_shapes
:         Щ
Tmodel_1/tf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0─
Omodel_1/tf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normalAdd[model_1/tf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/mul:z:0]model_1/tf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normal/mean:output:0*'
_output_shapes
:         *
T0Х
,model_1/tf_op_layer_Normal_1/sample/mul/onesConst*
_output_shapes
:*
dtype0*5
value,B*"   А?  А?  А?  А?  А?  А?  А?  А?А
;model_1/tf_op_layer_Normal_1/sample/mul/Normal_1/sample/mulMulSmodel_1/tf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normal:z:05model_1/tf_op_layer_Normal_1/sample/mul/ones:output:0*
T0*'
_output_shapes
:         Ц
-model_1/tf_op_layer_Normal_1/sample/add/zerosConst*5
value,B*"                                 *
_output_shapes
:*
dtype0я
;model_1/tf_op_layer_Normal_1/sample/add/Normal_1/sample/addAddV2?model_1/tf_op_layer_Normal_1/sample/mul/Normal_1/sample/mul:z:06model_1/tf_op_layer_Normal_1/sample/add/zeros:output:0*'
_output_shapes
:         *
T0Ы
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp4model_1_dense_3_matmul_readvariableop_dense_3_kernel*
dtype0*
_output_shapes
:	Ак
model_1/dense_3/MatMulMatMul'model_1/activation_1/Relu:activations:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Ц
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp3model_1_dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes
:*
dtype0ж
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ▓
Cmodel_1/tf_op_layer_Normal_1/sample/Shape_2/Normal_1/sample/Shape_2Shape?model_1/tf_op_layer_Normal_1/sample/add/Normal_1/sample/add:z:0*
T0*
_output_shapes
:К
Emodel_1/tf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/Minimum/yConst*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: Є
Cmodel_1/tf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/MinimumMinimum model_1/dense_3/BiasAdd:output:0Nmodel_1/tf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         ╡
kmodel_1/tf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0┼
emodel_1/tf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/ReshapeReshape8model_1/tf_op_layer_strided_slice/strided_slice:output:0tmodel_1/tf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/Reshape/shape:output:0*
T0*
_output_shapes
:Я
Umodel_1/tf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0б
Wmodel_1/tf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:б
Wmodel_1/tf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:У
Omodel_1/tf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_sliceStridedSliceLmodel_1/tf_op_layer_Normal_1/sample/Shape_2/Normal_1/sample/Shape_2:output:0^model_1/tf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack:output:0`model_1/tf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_1:output:0`model_1/tf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_2:output:0*
_output_shapes
:*
end_mask*
T0*
Index0z
5model_1/tf_op_layer_clip_by_value_1/clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *  а┴∙
3model_1/tf_op_layer_clip_by_value_1/clip_by_value_1MaximumGmodel_1/tf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/Minimum:z:0>model_1/tf_op_layer_clip_by_value_1/clip_by_value_1/y:output:0*'
_output_shapes
:         *
T0М
Jmodel_1/tf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : Ю
Emodel_1/tf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1ConcatV2nmodel_1/tf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/Reshape:output:0Xmodel_1/tf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice:output:0Smodel_1/tf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1/axis:output:0*
T0*
N*
_output_shapes
:С
Cmodel_1/tf_op_layer_Normal_1/sample/Reshape/Normal_1/sample/ReshapeReshape?model_1/tf_op_layer_Normal_1/sample/add/Normal_1/sample/add:z:0Nmodel_1/tf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1:output:0*'
_output_shapes
:         *
T0Н
model_1/tf_op_layer_Exp/ExpExp7model_1/tf_op_layer_clip_by_value_1/clip_by_value_1:z:0*'
_output_shapes
:         *
T0├
model_1/tf_op_layer_mul/mulMulLmodel_1/tf_op_layer_Normal_1/sample/Reshape/Normal_1/sample/Reshape:output:0model_1/tf_op_layer_Exp/Exp:y:0*'
_output_shapes
:         *
T0м
model_1/tf_op_layer_add/addAddV23model_1/tf_op_layer_clip_by_value/clip_by_value:z:0model_1/tf_op_layer_mul/mul:z:0*
T0*'
_output_shapes
:         x
model_1/tf_op_layer_Tanh/TanhTanhmodel_1/tf_op_layer_add/add:z:0*'
_output_shapes
:         *
T0С
IdentityIdentity!model_1/tf_op_layer_Tanh/Tanh:y:0%^model_1/dense/BiasAdd/ReadVariableOp$^model_1/dense/MatMul/ReadVariableOp'^model_1/dense_1/BiasAdd/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp'^model_1/dense_2/BiasAdd/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp5^model_1/layer_normalization/batchnorm/ReadVariableOp9^model_1/layer_normalization/batchnorm/mul/ReadVariableOp7^model_1/layer_normalization_1/batchnorm/ReadVariableOp;^model_1/layer_normalization_1/batchnorm/mul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::2t
8model_1/layer_normalization/batchnorm/mul/ReadVariableOp8model_1/layer_normalization/batchnorm/mul/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2x
:model_1/layer_normalization_1/batchnorm/mul/ReadVariableOp:model_1/layer_normalization_1/batchnorm/mul/ReadVariableOp2L
$model_1/dense/BiasAdd/ReadVariableOp$model_1/dense/BiasAdd/ReadVariableOp2J
#model_1/dense/MatMul/ReadVariableOp#model_1/dense/MatMul/ReadVariableOp2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2l
4model_1/layer_normalization/batchnorm/ReadVariableOp4model_1/layer_normalization/batchnorm/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp2p
6model_1/layer_normalization_1/batchnorm/ReadVariableOp6model_1/layer_normalization_1/batchnorm/ReadVariableOp:
 : : :. *
(
_user_specified_namepolicy_input_0: : : : : : : : :	 
└
H
,__inference_activation_1_layer_call_fn_73260

inputs
identityа
PartitionedCallPartitionedCallinputs*
Tin
2*,
_gradient_op_typePartitionedCall-71850*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*
Tout
2*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_71843a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
├
t
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_72367

inputs
inputs_1
identityP
addAddV2inputsinputs_1*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*9
_input_shapes(
&:         :         :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
╓
i
K__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_fn_73343
inputs_0
identity│
PartitionedCallPartitionedCallinputs_0*
Tout
2*
Tin
2*o
fjRh
f__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_and_return_conditional_losses_71998*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72005*
_output_shapes
:S
IdentityIdentityPartitionedCall:output:0*
_output_shapes
:*
T0"
identityIdentity:output:0*
_input_shapes
: :( $
"
_user_specified_name
inputs/0
▓
х
B__inference_dense_2_layer_call_and_return_conditional_losses_73270

inputs(
$matmul_readvariableop_dense_2_kernel'
#biasadd_readvariableop_dense_2_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
dtype0*
_output_shapes
:	Аi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
їЭ
д

B__inference_model_1_layer_call_and_return_conditional_losses_72700

inputs.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_biasI
Elayer_normalization_statefulpartitionedcall_layer_normalization_gammaH
Dlayer_normalization_statefulpartitionedcall_layer_normalization_beta2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_biasM
Ilayer_normalization_1_statefulpartitionedcall_layer_normalization_1_gammaL
Hlayer_normalization_1_statefulpartitionedcall_layer_normalization_1_beta2
.dense_2_statefulpartitionedcall_dense_2_kernel0
,dense_2_statefulpartitionedcall_dense_2_bias2
.dense_3_statefulpartitionedcall_dense_3_kernel0
,dense_3_statefulpartitionedcall_dense_3_bias
identityИвdense/StatefulPartitionedCallв,dense/bias/Regularizer/Square/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpвdense_1/StatefulPartitionedCallв.dense_1/bias/Regularizer/Square/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallвVtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallЗ
dense/StatefulPartitionedCallStatefulPartitionedCallinputs*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_71672*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-71679*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         А·
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0Elayer_normalization_statefulpartitionedcall_layer_normalization_gammaDlayer_normalization_statefulpartitionedcall_layer_normalization_beta*
Tout
2*,
_gradient_op_typePartitionedCall-71720*(
_output_shapes
:         А*W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_71713*
Tin
2*-
config_proto

GPU

CPU2*0J 8╫
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*
Tout
2*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_71737*,
_gradient_op_typePartitionedCall-71744*
Tin
2░
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         А*,
_gradient_op_typePartitionedCall-71785*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_71778И
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0Ilayer_normalization_1_statefulpartitionedcall_layer_normalization_1_gammaHlayer_normalization_1_statefulpartitionedcall_layer_normalization_1_beta*-
config_proto

GPU

CPU2*0J 8*Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_71819*
Tout
2*(
_output_shapes
:         А*,
_gradient_op_typePartitionedCall-71826*
Tin
2▌
activation_1/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
Tout
2*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_71843*,
_gradient_op_typePartitionedCall-71850▒
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0.dense_2_statefulpartitionedcall_dense_2_kernel,dense_2_statefulpartitionedcall_dense_2_bias*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_71868*
Tin
2*'
_output_shapes
:         *
Tout
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-71875°
1tf_op_layer_clip_by_value/Minimum/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-71900*e
f`R^
\__inference_tf_op_layer_clip_by_value/Minimum_layer_call_and_return_conditional_losses_71893*
Tin
2·
)tf_op_layer_clip_by_value/PartitionedCallPartitionedCall:tf_op_layer_clip_by_value/Minimum/PartitionedCall:output:0*
Tout
2*]
fXRV
T__inference_tf_op_layer_clip_by_value_layer_call_and_return_conditional_losses_71914*,
_gradient_op_typePartitionedCall-71921*
Tin
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         ╒
!tf_op_layer_Shape/PartitionedCallPartitionedCall2tf_op_layer_clip_by_value/PartitionedCall:output:0*
Tin
2*U
fPRN
L__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_71934*,
_gradient_op_typePartitionedCall-71941*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
:┘
)tf_op_layer_strided_slice/PartitionedCallPartitionedCall*tf_op_layer_Shape/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tin
2*
_output_shapes
: *,
_gradient_op_typePartitionedCall-71964*
Tout
2*]
fXRV
T__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_71957я
0tf_op_layer_Normal_1/sample/Prod/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*d
f_R]
[__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_and_return_conditional_losses_71978*,
_gradient_op_typePartitionedCall-71985*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
: *
Tout
2*
Tin
2Р
;tf_op_layer_Normal_1/sample/concat/values_0/PartitionedCallPartitionedCall9tf_op_layer_Normal_1/sample/Prod/PartitionedCall:output:0*
Tout
2*o
fjRh
f__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_and_return_conditional_losses_71998*,
_gradient_op_typePartitionedCall-72005*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
:Й
2tf_op_layer_Normal_1/sample/concat/PartitionedCallPartitionedCallDtf_op_layer_Normal_1/sample/concat/values_0/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tin
2*,
_gradient_op_typePartitionedCall-72027*
Tout
2*
_output_shapes
:*f
faR_
]__inference_tf_op_layer_Normal_1/sample/concat_layer_call_and_return_conditional_losses_72020▀
Vtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallStatefulPartitionedCall;tf_op_layer_Normal_1/sample/concat/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72047*
Tout
2*
Tin
2*0
_output_shapes
:                  *В
f}R{
y__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_and_return_conditional_losses_72040*-
config_proto

GPU

CPU2*0J 8╨
=tf_op_layer_Normal_1/sample/random_normal/mul/PartitionedCallPartitionedCall_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall:output:0*0
_output_shapes
:                  *
Tout
2*q
flRj
h__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_and_return_conditional_losses_72061*
Tin
2*,
_gradient_op_typePartitionedCall-72068*-
config_proto

GPU

CPU2*0J 8п
9tf_op_layer_Normal_1/sample/random_normal/PartitionedCallPartitionedCallFtf_op_layer_Normal_1/sample/random_normal/mul/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*m
fhRf
d__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_and_return_conditional_losses_72082*0
_output_shapes
:                  *
Tout
2*,
_gradient_op_typePartitionedCall-72089*
Tin
2О
/tf_op_layer_Normal_1/sample/mul/PartitionedCallPartitionedCallBtf_op_layer_Normal_1/sample/random_normal/PartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72110*
Tin
2*c
f^R\
Z__inference_tf_op_layer_Normal_1/sample/mul_layer_call_and_return_conditional_losses_72103*'
_output_shapes
:         Д
/tf_op_layer_Normal_1/sample/add/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/mul/PartitionedCall:output:0*'
_output_shapes
:         *
Tout
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72131*
Tin
2*c
f^R\
Z__inference_tf_op_layer_Normal_1/sample/add_layer_call_and_return_conditional_losses_72124▒
dense_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0.dense_3_statefulpartitionedcall_dense_3_kernel,dense_3_statefulpartitionedcall_dense_3_bias*,
_gradient_op_typePartitionedCall-72156*'
_output_shapes
:         *
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_72149 
3tf_op_layer_Normal_1/sample/Shape_2/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/add/PartitionedCall:output:0*
Tin
2*g
fbR`
^__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_and_return_conditional_losses_72173*
Tout
2*
_output_shapes
:*,
_gradient_op_typePartitionedCall-72180*-
config_proto

GPU

CPU2*0J 8№
3tf_op_layer_clip_by_value_1/Minimum/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tout
2*'
_output_shapes
:         *
Tin
2*,
_gradient_op_typePartitionedCall-72201*g
fbR`
^__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_and_return_conditional_losses_72194*-
config_proto

GPU

CPU2*0J 8Ы
Dtf_op_layer_Normal_1/sample/expand_to_vector/Reshape/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
Tout
2*
_output_shapes
:*,
_gradient_op_typePartitionedCall-72222*x
fsRq
o__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_and_return_conditional_losses_72215*-
config_proto

GPU

CPU2*0J 8*
Tin
2П
9tf_op_layer_Normal_1/sample/strided_slice/PartitionedCallPartitionedCall<tf_op_layer_Normal_1/sample/Shape_2/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
:*
Tin
2*,
_gradient_op_typePartitionedCall-72245*
Tout
2*m
fhRf
d__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_and_return_conditional_losses_72238А
+tf_op_layer_clip_by_value_1/PartitionedCallPartitionedCall<tf_op_layer_clip_by_value_1/Minimum/PartitionedCall:output:0*
Tin
2*'
_output_shapes
:         *_
fZRX
V__inference_tf_op_layer_clip_by_value_1_layer_call_and_return_conditional_losses_72259*
Tout
2*,
_gradient_op_typePartitionedCall-72266*-
config_proto

GPU

CPU2*0J 8█
4tf_op_layer_Normal_1/sample/concat_1/PartitionedCallPartitionedCallMtf_op_layer_Normal_1/sample/expand_to_vector/Reshape/PartitionedCall:output:0Btf_op_layer_Normal_1/sample/strided_slice/PartitionedCall:output:0*
_output_shapes
:*-
config_proto

GPU

CPU2*0J 8*
Tin
2*h
fcRa
___inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_and_return_conditional_losses_72281*
Tout
2*,
_gradient_op_typePartitionedCall-72289╒
3tf_op_layer_Normal_1/sample/Reshape/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/add/PartitionedCall:output:0=tf_op_layer_Normal_1/sample/concat_1/PartitionedCall:output:0*g
fbR`
^__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_and_return_conditional_losses_72303*
Tout
2*
Tin
2*0
_output_shapes
:                  *-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72311р
tf_op_layer_Exp/PartitionedCallPartitionedCall4tf_op_layer_clip_by_value_1/PartitionedCall:output:0*
Tin
2*,
_gradient_op_typePartitionedCall-72331*'
_output_shapes
:         *
Tout
2*S
fNRL
J__inference_tf_op_layer_Exp_layer_call_and_return_conditional_losses_72324*-
config_proto

GPU

CPU2*0J 8У
tf_op_layer_mul/PartitionedCallPartitionedCall<tf_op_layer_Normal_1/sample/Reshape/PartitionedCall:output:0(tf_op_layer_Exp/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72353*-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_72345*'
_output_shapes
:         *
Tin
2*
Tout
2Й
tf_op_layer_add/PartitionedCallPartitionedCall2tf_op_layer_clip_by_value/PartitionedCall:output:0(tf_op_layer_mul/PartitionedCall:output:0*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-72375*
Tout
2*
Tin
2*S
fNRL
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_72367*-
config_proto

GPU

CPU2*0J 8╓
 tf_op_layer_Tanh/PartitionedCallPartitionedCall(tf_op_layer_add/PartitionedCall:output:0*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72395*T
fORM
K__inference_tf_op_layer_Tanh_layer_call_and_return_conditional_losses_72388*
Tin
2*
Tout
2║
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_statefulpartitionedcall_dense_kernel^dense/StatefulPartitionedCall*
dtype0*
_output_shapes
:	АЛ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Аo
dense/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: С
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ▓
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_statefulpartitionedcall_dense_bias^dense/StatefulPartitionedCall*
_output_shapes	
:А*
dtype0Г
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0f
dense/bias/Regularizer/ConstConst*
_output_shapes
:*
valueB: *
dtype0М
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
_output_shapes
: *
T0a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5О
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0Л
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0├
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.dense_1_statefulpartitionedcall_dense_1_kernel ^dense_1/StatefulPartitionedCall* 
_output_shapes
:
АА*
dtype0Р
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААq
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ш
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *oГ:*
_output_shapes
: Ъ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Ч
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0║
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp,dense_1_statefulpartitionedcall_dense_1_bias ^dense_1/StatefulPartitionedCall*
dtype0*
_output_shapes	
:АЗ
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аh
dense_1/bias/Regularizer/ConstConst*
valueB: *
dtype0*
_output_shapes
:Т
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/mul/xConst*
valueB
 *╜7Ж5*
_output_shapes
: *
dtype0Ф
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: С
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0Є
IdentityIdentity)tf_op_layer_Tanh/PartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCallW^tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2░
Vtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallVtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: : : :	 :
 : : :& "
 
_user_specified_nameinputs: : : : : 
Ё
Т
'__inference_model_1_layer_call_fn_73084

inputs(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias5
1statefulpartitionedcall_layer_normalization_gamma4
0statefulpartitionedcall_layer_normalization_beta*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias7
3statefulpartitionedcall_layer_normalization_1_gamma6
2statefulpartitionedcall_layer_normalization_1_beta*
&statefulpartitionedcall_dense_2_kernel(
$statefulpartitionedcall_dense_2_bias*
&statefulpartitionedcall_dense_3_kernel(
$statefulpartitionedcall_dense_3_bias
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputs$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias1statefulpartitionedcall_layer_normalization_gamma0statefulpartitionedcall_layer_normalization_beta&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias3statefulpartitionedcall_layer_normalization_1_gamma2statefulpartitionedcall_layer_normalization_1_beta&statefulpartitionedcall_dense_2_kernel$statefulpartitionedcall_dense_2_bias&statefulpartitionedcall_dense_3_kernel$statefulpartitionedcall_dense_3_bias*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_72700*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72701*
Tin
2*
Tout
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :	 :
 : : :& "
 
_user_specified_nameinputs: 
е
v
Z__inference_tf_op_layer_Normal_1/sample/mul_layer_call_and_return_conditional_losses_72103

inputs
identitym
onesConst*
_output_shapes
:*5
value,B*"   А?  А?  А?  А?  А?  А?  А?  А?*
dtype0c
Normal_1/sample/mulMulinputsones:output:0*
T0*'
_output_shapes
:         _
IdentityIdentityNormal_1/sample/mul:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*/
_input_shapes
:                  :& "
 
_user_specified_nameinputs
К
r
V__inference_tf_op_layer_clip_by_value_1_layer_call_and_return_conditional_losses_72259

inputs
identityV
clip_by_value_1/yConst*
valueB
 *  а┴*
dtype0*
_output_shapes
: p
clip_by_value_1Maximuminputsclip_by_value_1/y:output:0*'
_output_shapes
:         *
T0[
IdentityIdentityclip_by_value_1:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
Ъ╛
п
B__inference_model_1_layer_call_and_return_conditional_losses_72910

inputs,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_biasN
Jlayer_normalization_batchnorm_mul_readvariableop_layer_normalization_gammaI
Elayer_normalization_batchnorm_readvariableop_layer_normalization_beta0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_biasR
Nlayer_normalization_1_batchnorm_mul_readvariableop_layer_normalization_1_gammaM
Ilayer_normalization_1_batchnorm_readvariableop_layer_normalization_1_beta0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias0
,dense_3_matmul_readvariableop_dense_3_kernel/
+dense_3_biasadd_readvariableop_dense_3_bias
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpв,dense/bias/Regularizer/Square/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв.dense_1/bias/Regularizer/Square/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpв,layer_normalization/batchnorm/ReadVariableOpв0layer_normalization/batchnorm/mul/ReadVariableOpв.layer_normalization_1/batchnorm/ReadVariableOpв2layer_normalization_1/batchnorm/mul/ReadVariableOpЕ
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes
:	Аv
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0Б
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0└
 layer_normalization/moments/meanMeandense/BiasAdd:output:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*
	keep_dims(*'
_output_shapes
:         Х
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*'
_output_shapes
:         └
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedense/BiasAdd:output:01layer_normalization/moments/StopGradient:output:0*(
_output_shapes
:         А*
T0А
6layer_normalization/moments/variance/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:у
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *oГ:╣
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         Е
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*'
_output_shapes
:         ╕
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpJlayer_normalization_batchnorm_mul_readvariableop_layer_normalization_gamma*
_output_shapes	
:А*
dtype0╛
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0Ь
#layer_normalization/batchnorm/mul_1Muldense/BiasAdd:output:0%layer_normalization/batchnorm/mul:z:0*(
_output_shapes
:         А*
T0п
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Ап
,layer_normalization/batchnorm/ReadVariableOpReadVariableOpElayer_normalization_batchnorm_readvariableop_layer_normalization_beta*
dtype0*
_output_shapes	
:А║
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         Ап
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аs
activation/ReluRelu'layer_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         А*
T0М
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
АА*
dtype0С
dense_1/MatMulMatMulactivation/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0З
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
dtype0*
_output_shapes	
:АП
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А~
4layer_normalization_1/moments/mean/reduction_indicesConst*
dtype0*
valueB:*
_output_shapes
:╞
"layer_normalization_1/moments/meanMeandense_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(Щ
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*'
_output_shapes
:         ╞
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:         АВ
8layer_normalization_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:щ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*'
_output_shapes
:         *
T0*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
dtype0*
valueB
 *oГ:*
_output_shapes
: ┐
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*'
_output_shapes
:         *
T0Й
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*'
_output_shapes
:         *
T0╛
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpNlayer_normalization_1_batchnorm_mul_readvariableop_layer_normalization_1_gamma*
dtype0*
_output_shapes	
:А─
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0в
%layer_normalization_1/batchnorm/mul_1Muldense_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*(
_output_shapes
:         А*
T0╡
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А╡
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpIlayer_normalization_1_batchnorm_readvariableop_layer_normalization_1_beta*
_output_shapes	
:А*
dtype0└
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         А╡
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*(
_output_shapes
:         А*
T0w
activation_1/ReluRelu)layer_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         АЛ
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
dtype0*
_output_shapes
:	АТ
dense_2/MatMulMatMulactivation_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ~
9tf_op_layer_clip_by_value/Minimum/clip_by_value/Minimum/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0╥
7tf_op_layer_clip_by_value/Minimum/clip_by_value/MinimumMinimumdense_2/BiasAdd:output:0Btf_op_layer_clip_by_value/Minimum/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:         n
)tf_op_layer_clip_by_value/clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *   └╒
'tf_op_layer_clip_by_value/clip_by_valueMaximum;tf_op_layer_clip_by_value/Minimum/clip_by_value/Minimum:z:02tf_op_layer_clip_by_value/clip_by_value/y:output:0*'
_output_shapes
:         *
T0r
tf_op_layer_Shape/ShapeShape+tf_op_layer_clip_by_value/clip_by_value:z:0*
_output_shapes
:*
T0w
-tf_op_layer_strided_slice/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0y
/tf_op_layer_strided_slice/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:y
/tf_op_layer_strided_slice/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:╦
'tf_op_layer_strided_slice/strided_sliceStridedSlice tf_op_layer_Shape/Shape:output:06tf_op_layer_strided_slice/strided_slice/stack:output:08tf_op_layer_strided_slice/strided_slice/stack_1:output:08tf_op_layer_strided_slice/strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0y
6tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/ConstConst*
dtype0*
_output_shapes
: *
valueB ╤
5tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/ProdProd0tf_op_layer_strided_slice/strided_slice:output:0?tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/Const:output:0*
T0*
_output_shapes
: ┴
Ktf_op_layer_Normal_1/sample/concat/values_0/Normal_1/sample/concat/values_0Pack>tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/Prod:output:0*
_output_shapes
:*
T0*
NК
@tf_op_layer_Normal_1/sample/concat/Normal_1/sample/BroadcastArgsConst*
valueB:*
dtype0*
_output_shapes
:А
>tf_op_layer_Normal_1/sample/concat/Normal_1/sample/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: ▌
9tf_op_layer_Normal_1/sample/concat/Normal_1/sample/concatConcatV2Ttf_op_layer_Normal_1/sample/concat/values_0/Normal_1/sample/concat/values_0:output:0Itf_op_layer_Normal_1/sample/concat/Normal_1/sample/BroadcastArgs:output:0Gtf_op_layer_Normal_1/sample/concat/Normal_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:М
qtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/Normal_1/sample/random_normal/RandomStandardNormalRandomStandardNormalBtf_op_layer_Normal_1/sample/concat/Normal_1/sample/concat:output:0*
dtype0*
T0*'
_output_shapes
:         Ч
Rtf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?с
Otf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/mulMulztf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/Normal_1/sample/random_normal/RandomStandardNormal:output:0[tf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/stddev:output:0*'
_output_shapes
:         *
T0С
Ltf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    м
Gtf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normalAddStf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/mul:z:0Utf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normal/mean:output:0*
T0*'
_output_shapes
:         Н
$tf_op_layer_Normal_1/sample/mul/onesConst*
dtype0*5
value,B*"   А?  А?  А?  А?  А?  А?  А?  А?*
_output_shapes
:ш
3tf_op_layer_Normal_1/sample/mul/Normal_1/sample/mulMulKtf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normal:z:0-tf_op_layer_Normal_1/sample/mul/ones:output:0*
T0*'
_output_shapes
:         О
%tf_op_layer_Normal_1/sample/add/zerosConst*
_output_shapes
:*5
value,B*"                                 *
dtype0╫
3tf_op_layer_Normal_1/sample/add/Normal_1/sample/addAddV27tf_op_layer_Normal_1/sample/mul/Normal_1/sample/mul:z:0.tf_op_layer_Normal_1/sample/add/zeros:output:0*'
_output_shapes
:         *
T0Л
dense_3/MatMul/ReadVariableOpReadVariableOp,dense_3_matmul_readvariableop_dense_3_kernel*
_output_shapes
:	А*
dtype0Т
dense_3/MatMulMatMulactivation_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Ж
dense_3/BiasAdd/ReadVariableOpReadVariableOp+dense_3_biasadd_readvariableop_dense_3_bias*
dtype0*
_output_shapes
:О
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         в
;tf_op_layer_Normal_1/sample/Shape_2/Normal_1/sample/Shape_2Shape7tf_op_layer_Normal_1/sample/add/Normal_1/sample/add:z:0*
_output_shapes
:*
T0В
=tf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/Minimum/yConst*
dtype0*
_output_shapes
: *
valueB
 *ЪЩЩ>┌
;tf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/MinimumMinimumdense_3/BiasAdd:output:0Ftf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:         *
T0н
ctf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0н
]tf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/ReshapeReshape0tf_op_layer_strided_slice/strided_slice:output:0ltf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/Reshape/shape:output:0*
T0*
_output_shapes
:Ч
Mtf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:Щ
Otf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_1Const*
dtype0*
valueB: *
_output_shapes
:Щ
Otf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:ы
Gtf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_sliceStridedSliceDtf_op_layer_Normal_1/sample/Shape_2/Normal_1/sample/Shape_2:output:0Vtf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack:output:0Xtf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_1:output:0Xtf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_2:output:0*
_output_shapes
:*
T0*
end_mask*
Index0r
-tf_op_layer_clip_by_value_1/clip_by_value_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *  а┴с
+tf_op_layer_clip_by_value_1/clip_by_value_1Maximum?tf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/Minimum:z:06tf_op_layer_clip_by_value_1/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         Д
Btf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : ■
=tf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1ConcatV2ftf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/Reshape:output:0Ptf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice:output:0Ktf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1/axis:output:0*
T0*
N*
_output_shapes
:∙
;tf_op_layer_Normal_1/sample/Reshape/Normal_1/sample/ReshapeReshape7tf_op_layer_Normal_1/sample/add/Normal_1/sample/add:z:0Ftf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1:output:0*'
_output_shapes
:         *
T0}
tf_op_layer_Exp/ExpExp/tf_op_layer_clip_by_value_1/clip_by_value_1:z:0*
T0*'
_output_shapes
:         л
tf_op_layer_mul/mulMulDtf_op_layer_Normal_1/sample/Reshape/Normal_1/sample/Reshape:output:0tf_op_layer_Exp/Exp:y:0*
T0*'
_output_shapes
:         Ф
tf_op_layer_add/addAddV2+tf_op_layer_clip_by_value/clip_by_value:z:0tf_op_layer_mul/mul:z:0*
T0*'
_output_shapes
:         h
tf_op_layer_Tanh/TanhTanhtf_op_layer_add/add:z:0*
T0*'
_output_shapes
:         ╢
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel^dense/MatMul/ReadVariableOp*
_output_shapes
:	А*
dtype0Л
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Аo
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: С
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ░
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias^dense/BiasAdd/ReadVariableOp*
_output_shapes	
:А*
dtype0Г
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0f
dense/bias/Regularizer/ConstConst*
dtype0*
valueB: *
_output_shapes
:М
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
_output_shapes
: *
T0a
dense/bias/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *╜7Ж5О
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Л
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0┐
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel^dense_1/MatMul/ReadVariableOp*
dtype0* 
_output_shapes
:
ААР
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААq
 dense_1/kernel/Regularizer/ConstConst*
valueB"       *
_output_shapes
:*
dtype0Ш
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *oГ:*
dtype0Ъ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ч
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0╕
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias^dense_1/BiasAdd/ReadVariableOp*
dtype0*
_output_shapes	
:АЗ
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:Аh
dense_1/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: Т
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: Ф
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    С
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: э
IdentityIdentitytf_op_layer_Tanh/Tanh:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
и
[
/__inference_tf_op_layer_mul_layer_call_fn_73529
inputs_0
inputs_1
identityп
PartitionedCallPartitionedCallinputs_0inputs_1*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-72353*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*S
fNRL
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_72345`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*B
_input_shapes1
/:                  :         :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
А
p
T__inference_tf_op_layer_clip_by_value_layer_call_and_return_conditional_losses_71914

inputs
identityT
clip_by_value/yConst*
dtype0*
valueB
 *   └*
_output_shapes
: l
clip_by_valueMaximuminputsclip_by_value/y:output:0*
T0*'
_output_shapes
:         Y
IdentityIdentityclip_by_value:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
х
а
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_71819

inputs<
8batchnorm_mul_readvariableop_layer_normalization_1_gamma7
3batchnorm_readvariableop_layer_normalization_1_beta
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0И
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*'
_output_shapes
:         *
T0*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:         И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*(
_output_shapes
:         А*
T0l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0з
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
	keep_dims(*
T0*'
_output_shapes
:         T
batchnorm/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: }
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*'
_output_shapes
:         ]
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*'
_output_shapes
:         Т
batchnorm/mul/ReadVariableOpReadVariableOp8batchnorm_mul_readvariableop_layer_normalization_1_gamma*
dtype0*
_output_shapes	
:АВ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*(
_output_shapes
:         А*
T0s
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*(
_output_shapes
:         АЙ
batchnorm/ReadVariableOpReadVariableOp3batchnorm_readvariableop_layer_normalization_1_beta*
dtype0*
_output_shapes	
:А~
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*(
_output_shapes
:         А*
T0Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Х
p
D__inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_fn_73484
inputs_0
inputs_1
identity╖
PartitionedCallPartitionedCallinputs_0inputs_1*h
fcRa
___inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_and_return_conditional_losses_72281*,
_gradient_op_typePartitionedCall-72289*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
:S
IdentityIdentityPartitionedCall:output:0*
_output_shapes
:*
T0"
identityIdentity:output:0*
_input_shapes
:::( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
▓
х
B__inference_dense_3_layer_call_and_return_conditional_losses_72149

inputs(
$matmul_readvariableop_dense_3_kernel'
#biasadd_readvariableop_dense_3_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_3_kernel*
dtype0*
_output_shapes
:	Аi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_3_bias*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*/
_input_shapes
:         А::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
Р
p
T__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_71957

inputs
identity]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0_
strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:╔
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
IdentityIdentitystrided_slice:output:0*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
::& "
 
_user_specified_nameinputs
Ш
g
I__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_fn_73387
inputs_0
identity╟
PartitionedCallPartitionedCallinputs_0*m
fhRf
d__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_and_return_conditional_losses_72082*
Tin
2*,
_gradient_op_typePartitionedCall-72089*0
_output_shapes
:                  *-
config_proto

GPU

CPU2*0J 8*
Tout
2i
IdentityIdentityPartitionedCall:output:0*0
_output_shapes
:                  *
T0"
identityIdentity:output:0*/
_input_shapes
:                  :( $
"
_user_specified_name
inputs/0
Ў
Н
o__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_and_return_conditional_losses_73442
inputs_0
identityx
.Normal_1/sample/expand_to_vector/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:Ы
(Normal_1/sample/expand_to_vector/ReshapeReshapeinputs_07Normal_1/sample/expand_to_vector/Reshape/shape:output:0*
_output_shapes
:*
T0l
IdentityIdentity1Normal_1/sample/expand_to_vector/Reshape:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*
_input_shapes
: :( $
"
_user_specified_name
inputs/0
°
i
K__inference_tf_op_layer_Tanh_layer_call_and_return_conditional_losses_73546
inputs_0
identityH
TanhTanhinputs_0*
T0*'
_output_shapes
:         P
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
√
c
G__inference_activation_1_layer_call_and_return_conditional_losses_71843

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:         А[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╩
t
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_72345

inputs
inputs_1
identityN
mulMulinputsinputs_1*
T0*'
_output_shapes
:         O
IdentityIdentitymul:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*B
_input_shapes1
/:                  :         :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
н
Д
f__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_and_return_conditional_losses_73338
inputs_0
identity_
Normal_1/sample/concat/values_0Packinputs_0*
N*
_output_shapes
:*
T0c
IdentityIdentity(Normal_1/sample/concat/values_0:output:0*
_output_shapes
:*
T0"
identityIdentity:output:0*
_input_shapes
: :( $
"
_user_specified_name
inputs/0
п
Ъ
y__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_and_return_conditional_losses_73360
inputs_0
identityИЬ
2Normal_1/sample/random_normal/RandomStandardNormalRandomStandardNormalinputs_0*0
_output_shapes
:                  *
dtype0*
T0М
IdentityIdentity;Normal_1/sample/random_normal/RandomStandardNormal:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*
_input_shapes
::( $
"
_user_specified_name
inputs/0
 
Д
h__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_and_return_conditional_losses_72061

inputs
identityi
$Normal_1/sample/random_normal/stddevConst*
_output_shapes
: *
valueB
 *  А?*
dtype0Ъ
!Normal_1/sample/random_normal/mulMulinputs-Normal_1/sample/random_normal/stddev:output:0*
T0*0
_output_shapes
:                  v
IdentityIdentity%Normal_1/sample/random_normal/mul:z:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*/
_input_shapes
:                  :& "
 
_user_specified_nameinputs
Э
|
^__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_and_return_conditional_losses_73414
inputs_0
identityO
Normal_1/sample/Shape_2Shapeinputs_0*
_output_shapes
:*
T0[
IdentityIdentity Normal_1/sample/Shape_2:output:0*
_output_shapes
:*
T0"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
Є
Л
___inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_and_return_conditional_losses_73478
inputs_0
inputs_1
identity_
Normal_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : О
Normal_1/sample/concat_1ConcatV2inputs_0inputs_1&Normal_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:\
IdentityIdentity!Normal_1/sample/concat_1:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*
_input_shapes
:::( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
Ч
z
^__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_and_return_conditional_losses_72173

inputs
identityM
Normal_1/sample/Shape_2Shapeinputs*
_output_shapes
:*
T0[
IdentityIdentity Normal_1/sample/Shape_2:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
ш
r
T__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_fn_73447
inputs_0
identity╝
PartitionedCallPartitionedCallinputs_0*
_output_shapes
:*,
_gradient_op_typePartitionedCall-72222*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*x
fsRq
o__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_and_return_conditional_losses_72215S
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*
_input_shapes
: :( $
"
_user_specified_name
inputs/0
Е
Ж
h__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_and_return_conditional_losses_73371
inputs_0
identityi
$Normal_1/sample/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ь
!Normal_1/sample/random_normal/mulMulinputs_0-Normal_1/sample/random_normal/stddev:output:0*
T0*0
_output_shapes
:                  v
IdentityIdentity%Normal_1/sample/random_normal/mul:z:0*0
_output_shapes
:                  *
T0"
identityIdentity:output:0*/
_input_shapes
:                  :( $
"
_user_specified_name
inputs/0
И
Ъ
'__inference_model_1_layer_call_fn_72716
policy_input_0(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias5
1statefulpartitionedcall_layer_normalization_gamma4
0statefulpartitionedcall_layer_normalization_beta*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias7
3statefulpartitionedcall_layer_normalization_1_gamma6
2statefulpartitionedcall_layer_normalization_1_beta*
&statefulpartitionedcall_dense_2_kernel(
$statefulpartitionedcall_dense_2_bias*
&statefulpartitionedcall_dense_3_kernel(
$statefulpartitionedcall_dense_3_bias
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallpolicy_input_0$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias1statefulpartitionedcall_layer_normalization_gamma0statefulpartitionedcall_layer_normalization_beta&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias3statefulpartitionedcall_layer_normalization_1_gamma2statefulpartitionedcall_layer_normalization_1_beta&statefulpartitionedcall_dense_2_kernel$statefulpartitionedcall_dense_2_bias&statefulpartitionedcall_dense_3_kernel$statefulpartitionedcall_dense_3_bias*-
config_proto

GPU

CPU2*0J 8*
Tout
2*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-72701*
Tin
2*K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_72700В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namepolicy_input_0: : : : : : : : :	 :
 : : 
я
А
d__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_and_return_conditional_losses_72082

inputs
identityg
"Normal_1/sample/random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    Ф
Normal_1/sample/random_normalAddinputs+Normal_1/sample/random_normal/mean:output:0*
T0*0
_output_shapes
:                  r
IdentityIdentity!Normal_1/sample/random_normal:z:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*/
_input_shapes
:                  :& "
 
_user_specified_nameinputs
╗
┘
3__inference_layer_normalization_layer_call_fn_73162

inputs5
1statefulpartitionedcall_layer_normalization_gamma4
0statefulpartitionedcall_layer_normalization_beta
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputs1statefulpartitionedcall_layer_normalization_gamma0statefulpartitionedcall_layer_normalization_beta*,
_gradient_op_typePartitionedCall-71720*
Tin
2*(
_output_shapes
:         А*
Tout
2*-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_71713Г
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
С3
╘
!__inference__traced_restore_73725
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias0
,assignvariableop_2_layer_normalization_gamma/
+assignvariableop_3_layer_normalization_beta%
!assignvariableop_4_dense_1_kernel#
assignvariableop_5_dense_1_bias2
.assignvariableop_6_layer_normalization_1_gamma1
-assignvariableop_7_layer_normalization_1_beta%
!assignvariableop_8_dense_2_kernel#
assignvariableop_9_dense_2_bias&
"assignvariableop_10_dense_3_kernel$
 assignvariableop_11_dense_3_bias
identity_13ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1Б
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*з
valueЭBЪB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0И
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*+
value"B B B B B B B B B B B B B *
dtype0┌
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2*D
_output_shapes2
0::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:y
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:}
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:М
AssignVariableOp_2AssignVariableOp,assignvariableop_2_layer_normalization_gammaIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0Л
AssignVariableOp_3AssignVariableOp+assignvariableop_3_layer_normalization_betaIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:Б
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0О
AssignVariableOp_6AssignVariableOp.assignvariableop_6_layer_normalization_1_gammaIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:Н
AssignVariableOp_7AssignVariableOp-assignvariableop_7_layer_normalization_1_betaIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0Б
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0Д
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_3_kernelIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0В
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_3_biasIdentity_11:output:0*
_output_shapes
 *
dtype0М
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0╡
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ╫
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ф
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : 
Ф
К
^__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_and_return_conditional_losses_73501
inputs_0
inputs_1
identityq
Normal_1/sample/ReshapeReshapeinputs_0inputs_1*
T0*0
_output_shapes
:                  q
IdentityIdentity Normal_1/sample/Reshape:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*,
_input_shapes
:         ::( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
Ў
а
__inference_loss_fn_3_73612?
;dense_1_bias_regularizer_square_readvariableop_dense_1_bias
identityИв.dense_1/bias/Regularizer/Square/ReadVariableOpз
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp;dense_1_bias_regularizer_square_readvariableop_dense_1_bias*
dtype0*
_output_shapes	
:АЗ
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0h
dense_1/bias/Regularizer/ConstConst*
dtype0*
valueB: *
_output_shapes
:Т
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/mul/xConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: Ф
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0С
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0И
IdentityIdentity dense_1/bias/Regularizer/add:z:0/^dense_1/bias/Regularizer/Square/ReadVariableOp*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
:2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp:  
▌
W
9__inference_tf_op_layer_clip_by_value_layer_call_fn_73299
inputs_0
identityо
PartitionedCallPartitionedCallinputs_0*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-71921*]
fXRV
T__inference_tf_op_layer_clip_by_value_layer_call_and_return_conditional_losses_71914*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         `
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
╕
^
@__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_fn_73333
inputs_0
identityд
PartitionedCallPartitionedCallinputs_0*,
_gradient_op_typePartitionedCall-71985*
_output_shapes
: *
Tout
2*
Tin
2*d
f_R]
[__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_and_return_conditional_losses_71978*-
config_proto

GPU

CPU2*0J 8O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
: :( $
"
_user_specified_name
inputs/0
х
а
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_73243

inputs<
8batchnorm_mul_readvariableop_layer_normalization_1_gamma7
3batchnorm_readvariableop_layer_normalization_1_beta
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:И
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*'
_output_shapes
:         *
	keep_dims(*
T0m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:         И
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Аl
"moments/variance/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0з
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
	keep_dims(*'
_output_shapes
:         *
T0T
batchnorm/add/yConst*
valueB
 *oГ:*
_output_shapes
: *
dtype0}
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*'
_output_shapes
:         *
T0]
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*'
_output_shapes
:         *
T0Т
batchnorm/mul/ReadVariableOpReadVariableOp8batchnorm_mul_readvariableop_layer_normalization_1_gamma*
dtype0*
_output_shapes	
:АВ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Аs
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*(
_output_shapes
:         А*
T0Й
batchnorm/ReadVariableOpReadVariableOp3batchnorm_readvariableop_layer_normalization_1_beta*
_output_shapes	
:А*
dtype0~
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         Аs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         АЦ
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
╓
g
I__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_fn_73460
inputs_0
identity▒
PartitionedCallPartitionedCallinputs_0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
:*m
fhRf
d__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_and_return_conditional_losses_72238*
Tin
2*,
_gradient_op_typePartitionedCall-72245S
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*
_input_shapes
::( $
"
_user_specified_name
inputs/0
Ц
r
T__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_73317
inputs_0
identity]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0╦
strided_sliceStridedSliceinputs_0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
shrink_axis_mask*
_output_shapes
: *
T0M
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
::( $
"
_user_specified_name
inputs/0
щ
]
?__inference_tf_op_layer_Normal_1/sample/add_layer_call_fn_73409
inputs_0
identity┤
PartitionedCallPartitionedCallinputs_0*
Tout
2*,
_gradient_op_typePartitionedCall-72131*c
f^R\
Z__inference_tf_op_layer_Normal_1/sample/add_layer_call_and_return_conditional_losses_72124*
Tin
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         `
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
Ъ╛
п
B__inference_model_1_layer_call_and_return_conditional_losses_73050

inputs,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_biasN
Jlayer_normalization_batchnorm_mul_readvariableop_layer_normalization_gammaI
Elayer_normalization_batchnorm_readvariableop_layer_normalization_beta0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_biasR
Nlayer_normalization_1_batchnorm_mul_readvariableop_layer_normalization_1_gammaM
Ilayer_normalization_1_batchnorm_readvariableop_layer_normalization_1_beta0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias0
,dense_3_matmul_readvariableop_dense_3_kernel/
+dense_3_biasadd_readvariableop_dense_3_bias
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpв,dense/bias/Regularizer/Square/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв.dense_1/bias/Regularizer/Square/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpв,layer_normalization/batchnorm/ReadVariableOpв0layer_normalization/batchnorm/mul/ReadVariableOpв.layer_normalization_1/batchnorm/ReadVariableOpв2layer_normalization_1/batchnorm/mul/ReadVariableOpЕ
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
dtype0*
_output_shapes
:	Аv
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АБ
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:А*
dtype0Й
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:└
 layer_normalization/moments/meanMeandense/BiasAdd:output:0;layer_normalization/moments/mean/reduction_indices:output:0*
	keep_dims(*
T0*'
_output_shapes
:         Х
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*'
_output_shapes
:         *
T0└
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedense/BiasAdd:output:01layer_normalization/moments/StopGradient:output:0*(
_output_shapes
:         А*
T0А
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:у
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*'
_output_shapes
:         *
T0*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *oГ:*
dtype0╣
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         Е
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*'
_output_shapes
:         ╕
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpJlayer_normalization_batchnorm_mul_readvariableop_layer_normalization_gamma*
dtype0*
_output_shapes	
:А╛
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0Ь
#layer_normalization/batchnorm/mul_1Muldense/BiasAdd:output:0%layer_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Ап
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Ап
,layer_normalization/batchnorm/ReadVariableOpReadVariableOpElayer_normalization_batchnorm_readvariableop_layer_normalization_beta*
dtype0*
_output_shapes	
:А║
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*(
_output_shapes
:         А*
T0п
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*(
_output_shapes
:         А*
T0s
activation/ReluRelu'layer_normalization/batchnorm/add_1:z:0*(
_output_shapes
:         А*
T0М
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
АА*
dtype0С
dense_1/MatMulMatMulactivation/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0З
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:А*
dtype0П
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А~
4layer_normalization_1/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:╞
"layer_normalization_1/moments/meanMeandense_1/BiasAdd:output:0=layer_normalization_1/moments/mean/reduction_indices:output:0*'
_output_shapes
:         *
T0*
	keep_dims(Щ
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*'
_output_shapes
:         ╞
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencedense_1/BiasAdd:output:03layer_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:         АВ
8layer_normalization_1/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:щ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
	keep_dims(*'
_output_shapes
:         *
T0j
%layer_normalization_1/batchnorm/add/yConst*
dtype0*
valueB
 *oГ:*
_output_shapes
: ┐
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*'
_output_shapes
:         *
T0Й
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*'
_output_shapes
:         ╛
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpNlayer_normalization_1_batchnorm_mul_readvariableop_layer_normalization_1_gamma*
dtype0*
_output_shapes	
:А─
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ав
%layer_normalization_1/batchnorm/mul_1Muldense_1/BiasAdd:output:0'layer_normalization_1/batchnorm/mul:z:0*(
_output_shapes
:         А*
T0╡
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*(
_output_shapes
:         А*
T0╡
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpIlayer_normalization_1_batchnorm_readvariableop_layer_normalization_1_beta*
_output_shapes	
:А*
dtype0└
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*(
_output_shapes
:         А*
T0╡
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         Аw
activation_1/ReluRelu)layer_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         АЛ
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
dtype0*
_output_shapes
:	АТ
dense_2/MatMulMatMulactivation_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Ж
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ~
9tf_op_layer_clip_by_value/Minimum/clip_by_value/Minimum/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: ╥
7tf_op_layer_clip_by_value/Minimum/clip_by_value/MinimumMinimumdense_2/BiasAdd:output:0Btf_op_layer_clip_by_value/Minimum/clip_by_value/Minimum/y:output:0*'
_output_shapes
:         *
T0n
)tf_op_layer_clip_by_value/clip_by_value/yConst*
dtype0*
_output_shapes
: *
valueB
 *   └╒
'tf_op_layer_clip_by_value/clip_by_valueMaximum;tf_op_layer_clip_by_value/Minimum/clip_by_value/Minimum:z:02tf_op_layer_clip_by_value/clip_by_value/y:output:0*
T0*'
_output_shapes
:         r
tf_op_layer_Shape/ShapeShape+tf_op_layer_clip_by_value/clip_by_value:z:0*
_output_shapes
:*
T0w
-tf_op_layer_strided_slice/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:y
/tf_op_layer_strided_slice/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:y
/tf_op_layer_strided_slice/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0╦
'tf_op_layer_strided_slice/strided_sliceStridedSlice tf_op_layer_Shape/Shape:output:06tf_op_layer_strided_slice/strided_slice/stack:output:08tf_op_layer_strided_slice/strided_slice/stack_1:output:08tf_op_layer_strided_slice/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
6tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB ╤
5tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/ProdProd0tf_op_layer_strided_slice/strided_slice:output:0?tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/Const:output:0*
T0*
_output_shapes
: ┴
Ktf_op_layer_Normal_1/sample/concat/values_0/Normal_1/sample/concat/values_0Pack>tf_op_layer_Normal_1/sample/Prod/Normal_1/sample/Prod:output:0*
N*
_output_shapes
:*
T0К
@tf_op_layer_Normal_1/sample/concat/Normal_1/sample/BroadcastArgsConst*
dtype0*
valueB:*
_output_shapes
:А
>tf_op_layer_Normal_1/sample/concat/Normal_1/sample/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0▌
9tf_op_layer_Normal_1/sample/concat/Normal_1/sample/concatConcatV2Ttf_op_layer_Normal_1/sample/concat/values_0/Normal_1/sample/concat/values_0:output:0Itf_op_layer_Normal_1/sample/concat/Normal_1/sample/BroadcastArgs:output:0Gtf_op_layer_Normal_1/sample/concat/Normal_1/sample/concat/axis:output:0*
_output_shapes
:*
N*
T0М
qtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/Normal_1/sample/random_normal/RandomStandardNormalRandomStandardNormalBtf_op_layer_Normal_1/sample/concat/Normal_1/sample/concat:output:0*
T0*
dtype0*'
_output_shapes
:         Ч
Rtf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?с
Otf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/mulMulztf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/Normal_1/sample/random_normal/RandomStandardNormal:output:0[tf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/stddev:output:0*
T0*'
_output_shapes
:         С
Ltf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    м
Gtf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normalAddStf_op_layer_Normal_1/sample/random_normal/mul/Normal_1/sample/random_normal/mul:z:0Utf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normal/mean:output:0*
T0*'
_output_shapes
:         Н
$tf_op_layer_Normal_1/sample/mul/onesConst*5
value,B*"   А?  А?  А?  А?  А?  А?  А?  А?*
dtype0*
_output_shapes
:ш
3tf_op_layer_Normal_1/sample/mul/Normal_1/sample/mulMulKtf_op_layer_Normal_1/sample/random_normal/Normal_1/sample/random_normal:z:0-tf_op_layer_Normal_1/sample/mul/ones:output:0*
T0*'
_output_shapes
:         О
%tf_op_layer_Normal_1/sample/add/zerosConst*5
value,B*"                                 *
_output_shapes
:*
dtype0╫
3tf_op_layer_Normal_1/sample/add/Normal_1/sample/addAddV27tf_op_layer_Normal_1/sample/mul/Normal_1/sample/mul:z:0.tf_op_layer_Normal_1/sample/add/zeros:output:0*'
_output_shapes
:         *
T0Л
dense_3/MatMul/ReadVariableOpReadVariableOp,dense_3_matmul_readvariableop_dense_3_kernel*
dtype0*
_output_shapes
:	АТ
dense_3/MatMulMatMulactivation_1/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ж
dense_3/BiasAdd/ReadVariableOpReadVariableOp+dense_3_biasadd_readvariableop_dense_3_bias*
_output_shapes
:*
dtype0О
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         в
;tf_op_layer_Normal_1/sample/Shape_2/Normal_1/sample/Shape_2Shape7tf_op_layer_Normal_1/sample/add/Normal_1/sample/add:z:0*
_output_shapes
:*
T0В
=tf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/Minimum/yConst*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: ┌
;tf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/MinimumMinimumdense_3/BiasAdd:output:0Ftf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:         *
T0н
ctf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:н
]tf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/ReshapeReshape0tf_op_layer_strided_slice/strided_slice:output:0ltf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/Reshape/shape:output:0*
_output_shapes
:*
T0Ч
Mtf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0Щ
Otf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Щ
Otf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:ы
Gtf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_sliceStridedSliceDtf_op_layer_Normal_1/sample/Shape_2/Normal_1/sample/Shape_2:output:0Vtf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack:output:0Xtf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_1:output:0Xtf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
-tf_op_layer_clip_by_value_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  а┴с
+tf_op_layer_clip_by_value_1/clip_by_value_1Maximum?tf_op_layer_clip_by_value_1/Minimum/clip_by_value_1/Minimum:z:06tf_op_layer_clip_by_value_1/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:         Д
Btf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0■
=tf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1ConcatV2ftf_op_layer_Normal_1/sample/expand_to_vector/Reshape/Normal_1/sample/expand_to_vector/Reshape:output:0Ptf_op_layer_Normal_1/sample/strided_slice/Normal_1/sample/strided_slice:output:0Ktf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1/axis:output:0*
_output_shapes
:*
N*
T0∙
;tf_op_layer_Normal_1/sample/Reshape/Normal_1/sample/ReshapeReshape7tf_op_layer_Normal_1/sample/add/Normal_1/sample/add:z:0Ftf_op_layer_Normal_1/sample/concat_1/Normal_1/sample/concat_1:output:0*'
_output_shapes
:         *
T0}
tf_op_layer_Exp/ExpExp/tf_op_layer_clip_by_value_1/clip_by_value_1:z:0*
T0*'
_output_shapes
:         л
tf_op_layer_mul/mulMulDtf_op_layer_Normal_1/sample/Reshape/Normal_1/sample/Reshape:output:0tf_op_layer_Exp/Exp:y:0*'
_output_shapes
:         *
T0Ф
tf_op_layer_add/addAddV2+tf_op_layer_clip_by_value/clip_by_value:z:0tf_op_layer_mul/mul:z:0*
T0*'
_output_shapes
:         h
tf_op_layer_Tanh/TanhTanhtf_op_layer_add/add:z:0*
T0*'
_output_shapes
:         ╢
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel^dense/MatMul/ReadVariableOp*
dtype0*
_output_shapes
:	АЛ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes
:	А*
T0o
dense/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/mul/xConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0С
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0░
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias^dense/BiasAdd/ReadVariableOp*
_output_shapes	
:А*
dtype0Г
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0f
dense/bias/Regularizer/ConstConst*
valueB: *
_output_shapes
:*
dtype0М
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *╜7Ж5*
dtype0О
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
_output_shapes
: *
T0a
dense/bias/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    Л
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0┐
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel^dense_1/MatMul/ReadVariableOp*
dtype0* 
_output_shapes
:
ААР
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0* 
_output_shapes
:
АА*
T0q
 dense_1/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       Ш
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
valueB
 *oГ:*
dtype0Ъ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/add/xConst*
dtype0*
_output_shapes
: *
valueB
 *    Ч
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ╕
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias^dense_1/BiasAdd/ReadVariableOp*
_output_shapes	
:А*
dtype0З
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0h
dense_1/bias/Regularizer/ConstConst*
valueB: *
_output_shapes
:*
dtype0Т
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/mul/xConst*
valueB
 *╜7Ж5*
_output_shapes
: *
dtype0Ф
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0С
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: э
IdentityIdentitytf_op_layer_Tanh/Tanh:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
ъ
Й
___inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_and_return_conditional_losses_72281

inputs
inputs_1
identity_
Normal_1/sample/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0М
Normal_1/sample/concat_1ConcatV2inputsinputs_1&Normal_1/sample/concat_1/axis:output:0*
T0*
_output_shapes
:*
N\
IdentityIdentity!Normal_1/sample/concat_1:output:0*
_output_shapes
:*
T0"
identityIdentity:output:0*
_input_shapes
:::& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
є
╢
'__inference_dense_3_layer_call_fn_73436

inputs*
&statefulpartitionedcall_dense_3_kernel(
$statefulpartitionedcall_dense_3_bias
identityИвStatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputs&statefulpartitionedcall_dense_3_kernel$statefulpartitionedcall_dense_3_bias*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_72149*,
_gradient_op_typePartitionedCall-72156*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
б
А
d__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_and_return_conditional_losses_72238

inputs
identitym
#Normal_1/sample/strided_slice/stackConst*
dtype0*
valueB:*
_output_shapes
:o
%Normal_1/sample/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:o
%Normal_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
Normal_1/sample/strided_sliceStridedSliceinputs,Normal_1/sample/strided_slice/stack:output:0.Normal_1/sample/strided_slice/stack_1:output:0.Normal_1/sample/strided_slice/stack_2:output:0*
_output_shapes
:*
Index0*
end_mask*
T0a
IdentityIdentity&Normal_1/sample/strided_slice:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*
_input_shapes
::& "
 
_user_specified_nameinputs
╔
M
/__inference_tf_op_layer_Exp_layer_call_fn_73517
inputs_0
identityд
PartitionedCallPartitionedCallinputs_0*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-72331*S
fNRL
J__inference_tf_op_layer_Exp_layer_call_and_return_conditional_losses_72324*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
з
В
d__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_and_return_conditional_losses_73455
inputs_0
identitym
#Normal_1/sample/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0o
%Normal_1/sample/strided_slice/stack_1Const*
valueB: *
_output_shapes
:*
dtype0o
%Normal_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
Normal_1/sample/strided_sliceStridedSliceinputs_0,Normal_1/sample/strided_slice/stack:output:0.Normal_1/sample/strided_slice/stack_1:output:0.Normal_1/sample/strided_slice/stack_2:output:0*
_output_shapes
:*
Index0*
end_mask*
T0a
IdentityIdentity&Normal_1/sample/strided_slice:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*
_input_shapes
::( $
"
_user_specified_name
inputs/0
НЮ
м

B__inference_model_1_layer_call_and_return_conditional_losses_72518
policy_input_0.
*dense_statefulpartitionedcall_dense_kernel,
(dense_statefulpartitionedcall_dense_biasI
Elayer_normalization_statefulpartitionedcall_layer_normalization_gammaH
Dlayer_normalization_statefulpartitionedcall_layer_normalization_beta2
.dense_1_statefulpartitionedcall_dense_1_kernel0
,dense_1_statefulpartitionedcall_dense_1_biasM
Ilayer_normalization_1_statefulpartitionedcall_layer_normalization_1_gammaL
Hlayer_normalization_1_statefulpartitionedcall_layer_normalization_1_beta2
.dense_2_statefulpartitionedcall_dense_2_kernel0
,dense_2_statefulpartitionedcall_dense_2_bias2
.dense_3_statefulpartitionedcall_dense_3_kernel0
,dense_3_statefulpartitionedcall_dense_3_bias
identityИвdense/StatefulPartitionedCallв,dense/bias/Regularizer/Square/ReadVariableOpв.dense/kernel/Regularizer/Square/ReadVariableOpвdense_1/StatefulPartitionedCallв.dense_1/bias/Regularizer/Square/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOpвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallв+layer_normalization/StatefulPartitionedCallв-layer_normalization_1/StatefulPartitionedCallвVtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallП
dense/StatefulPartitionedCallStatefulPartitionedCallpolicy_input_0*dense_statefulpartitionedcall_dense_kernel(dense_statefulpartitionedcall_dense_bias*(
_output_shapes
:         А*,
_gradient_op_typePartitionedCall-71679*
Tin
2*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_71672*
Tout
2·
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0Elayer_normalization_statefulpartitionedcall_layer_normalization_gammaDlayer_normalization_statefulpartitionedcall_layer_normalization_beta*,
_gradient_op_typePartitionedCall-71720*W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_71713*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         А╫
activation/PartitionedCallPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0*
Tout
2*,
_gradient_op_typePartitionedCall-71744*(
_output_shapes
:         А*
Tin
2*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_71737░
dense_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0.dense_1_statefulpartitionedcall_dense_1_kernel,dense_1_statefulpartitionedcall_dense_1_bias*,
_gradient_op_typePartitionedCall-71785*
Tin
2*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_71778*-
config_proto

GPU

CPU2*0J 8*
Tout
2*(
_output_shapes
:         АИ
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0Ilayer_normalization_1_statefulpartitionedcall_layer_normalization_1_gammaHlayer_normalization_1_statefulpartitionedcall_layer_normalization_1_beta*Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_71819*
Tin
2*(
_output_shapes
:         А*,
_gradient_op_typePartitionedCall-71826*
Tout
2*-
config_proto

GPU

CPU2*0J 8▌
activation_1/PartitionedCallPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-71850*
Tin
2*-
config_proto

GPU

CPU2*0J 8*
Tout
2*(
_output_shapes
:         А*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_71843▒
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0.dense_2_statefulpartitionedcall_dense_2_kernel,dense_2_statefulpartitionedcall_dense_2_bias*'
_output_shapes
:         *
Tout
2*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_71868*,
_gradient_op_typePartitionedCall-71875*
Tin
2°
1tf_op_layer_clip_by_value/Minimum/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tout
2*
Tin
2*e
f`R^
\__inference_tf_op_layer_clip_by_value/Minimum_layer_call_and_return_conditional_losses_71893*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-71900*'
_output_shapes
:         ·
)tf_op_layer_clip_by_value/PartitionedCallPartitionedCall:tf_op_layer_clip_by_value/Minimum/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tout
2*,
_gradient_op_typePartitionedCall-71921*'
_output_shapes
:         *]
fXRV
T__inference_tf_op_layer_clip_by_value_layer_call_and_return_conditional_losses_71914*
Tin
2╒
!tf_op_layer_Shape/PartitionedCallPartitionedCall2tf_op_layer_clip_by_value/PartitionedCall:output:0*U
fPRN
L__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_71934*
_output_shapes
:*,
_gradient_op_typePartitionedCall-71941*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8┘
)tf_op_layer_strided_slice/PartitionedCallPartitionedCall*tf_op_layer_Shape/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-71964*
Tin
2*
Tout
2*]
fXRV
T__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_71957*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
: я
0tf_op_layer_Normal_1/sample/Prod/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*
Tout
2*d
f_R]
[__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_and_return_conditional_losses_71978*,
_gradient_op_typePartitionedCall-71985*
Tin
2Р
;tf_op_layer_Normal_1/sample/concat/values_0/PartitionedCallPartitionedCall9tf_op_layer_Normal_1/sample/Prod/PartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*o
fjRh
f__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_and_return_conditional_losses_71998*,
_gradient_op_typePartitionedCall-72005*
_output_shapes
:Й
2tf_op_layer_Normal_1/sample/concat/PartitionedCallPartitionedCallDtf_op_layer_Normal_1/sample/concat/values_0/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72027*
_output_shapes
:*
Tout
2*
Tin
2*f
faR_
]__inference_tf_op_layer_Normal_1/sample/concat_layer_call_and_return_conditional_losses_72020*-
config_proto

GPU

CPU2*0J 8▀
Vtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallStatefulPartitionedCall;tf_op_layer_Normal_1/sample/concat/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72047*
Tout
2*
Tin
2*В
f}R{
y__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_and_return_conditional_losses_72040*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:                  ╨
=tf_op_layer_Normal_1/sample/random_normal/mul/PartitionedCallPartitionedCall_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall:output:0*0
_output_shapes
:                  *q
flRj
h__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_and_return_conditional_losses_72061*
Tin
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72068*
Tout
2п
9tf_op_layer_Normal_1/sample/random_normal/PartitionedCallPartitionedCallFtf_op_layer_Normal_1/sample/random_normal/mul/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-72089*
Tin
2*m
fhRf
d__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_and_return_conditional_losses_72082*-
config_proto

GPU

CPU2*0J 8*
Tout
2*0
_output_shapes
:                  О
/tf_op_layer_Normal_1/sample/mul/PartitionedCallPartitionedCallBtf_op_layer_Normal_1/sample/random_normal/PartitionedCall:output:0*c
f^R\
Z__inference_tf_op_layer_Normal_1/sample/mul_layer_call_and_return_conditional_losses_72103*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-72110*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
Tin
2Д
/tf_op_layer_Normal_1/sample/add/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/mul/PartitionedCall:output:0*'
_output_shapes
:         *
Tout
2*,
_gradient_op_typePartitionedCall-72131*c
f^R\
Z__inference_tf_op_layer_Normal_1/sample/add_layer_call_and_return_conditional_losses_72124*
Tin
2*-
config_proto

GPU

CPU2*0J 8▒
dense_3/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0.dense_3_statefulpartitionedcall_dense_3_kernel,dense_3_statefulpartitionedcall_dense_3_bias*K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_72149*
Tin
2*,
_gradient_op_typePartitionedCall-72156*'
_output_shapes
:         *
Tout
2*-
config_proto

GPU

CPU2*0J 8 
3tf_op_layer_Normal_1/sample/Shape_2/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/add/PartitionedCall:output:0*
Tin
2*g
fbR`
^__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_and_return_conditional_losses_72173*,
_gradient_op_typePartitionedCall-72180*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
:№
3tf_op_layer_clip_by_value_1/Minimum/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tout
2*g
fbR`
^__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_and_return_conditional_losses_72194*
Tin
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-72201Ы
Dtf_op_layer_Normal_1/sample/expand_to_vector/Reshape/PartitionedCallPartitionedCall2tf_op_layer_strided_slice/PartitionedCall:output:0*
Tin
2*x
fsRq
o__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_and_return_conditional_losses_72215*
_output_shapes
:*
Tout
2*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72222П
9tf_op_layer_Normal_1/sample/strided_slice/PartitionedCallPartitionedCall<tf_op_layer_Normal_1/sample/Shape_2/PartitionedCall:output:0*m
fhRf
d__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_and_return_conditional_losses_72238*,
_gradient_op_typePartitionedCall-72245*-
config_proto

GPU

CPU2*0J 8*
Tout
2*
_output_shapes
:*
Tin
2А
+tf_op_layer_clip_by_value_1/PartitionedCallPartitionedCall<tf_op_layer_clip_by_value_1/Minimum/PartitionedCall:output:0*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-72266*
Tin
2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*_
fZRX
V__inference_tf_op_layer_clip_by_value_1_layer_call_and_return_conditional_losses_72259█
4tf_op_layer_Normal_1/sample/concat_1/PartitionedCallPartitionedCallMtf_op_layer_Normal_1/sample/expand_to_vector/Reshape/PartitionedCall:output:0Btf_op_layer_Normal_1/sample/strided_slice/PartitionedCall:output:0*h
fcRa
___inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_and_return_conditional_losses_72281*-
config_proto

GPU

CPU2*0J 8*
Tout
2*,
_gradient_op_typePartitionedCall-72289*
Tin
2*
_output_shapes
:╒
3tf_op_layer_Normal_1/sample/Reshape/PartitionedCallPartitionedCall8tf_op_layer_Normal_1/sample/add/PartitionedCall:output:0=tf_op_layer_Normal_1/sample/concat_1/PartitionedCall:output:0*
Tin
2*0
_output_shapes
:                  *,
_gradient_op_typePartitionedCall-72311*g
fbR`
^__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_and_return_conditional_losses_72303*
Tout
2*-
config_proto

GPU

CPU2*0J 8р
tf_op_layer_Exp/PartitionedCallPartitionedCall4tf_op_layer_clip_by_value_1/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72331*S
fNRL
J__inference_tf_op_layer_Exp_layer_call_and_return_conditional_losses_72324*'
_output_shapes
:         *
Tout
2*
Tin
2У
tf_op_layer_mul/PartitionedCallPartitionedCall<tf_op_layer_Normal_1/sample/Reshape/PartitionedCall:output:0(tf_op_layer_Exp/PartitionedCall:output:0*
Tin
2*,
_gradient_op_typePartitionedCall-72353*
Tout
2*S
fNRL
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_72345*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         Й
tf_op_layer_add/PartitionedCallPartitionedCall2tf_op_layer_clip_by_value/PartitionedCall:output:0(tf_op_layer_mul/PartitionedCall:output:0*
Tout
2*S
fNRL
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_72367*,
_gradient_op_typePartitionedCall-72375*'
_output_shapes
:         *
Tin
2*-
config_proto

GPU

CPU2*0J 8╓
 tf_op_layer_Tanh/PartitionedCallPartitionedCall(tf_op_layer_add/PartitionedCall:output:0*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*
Tin
2*,
_gradient_op_typePartitionedCall-72395*T
fORM
K__inference_tf_op_layer_Tanh_layer_call_and_return_conditional_losses_72388*
Tout
2║
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_statefulpartitionedcall_dense_kernel^dense/StatefulPartitionedCall*
dtype0*
_output_shapes
:	АЛ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	Аo
dense/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
valueB
 *    *
dtype0С
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ▓
,dense/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_statefulpartitionedcall_dense_bias^dense/StatefulPartitionedCall*
dtype0*
_output_shapes	
:АГ
dense/bias/Regularizer/SquareSquare4dense/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0f
dense/bias/Regularizer/ConstConst*
dtype0*
valueB: *
_output_shapes
:М
dense/bias/Regularizer/SumSum!dense/bias/Regularizer/Square:y:0%dense/bias/Regularizer/Const:output:0*
_output_shapes
: *
T0a
dense/bias/Regularizer/mul/xConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: О
dense/bias/Regularizer/mulMul%dense/bias/Regularizer/mul/x:output:0#dense/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
dense/bias/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: Л
dense/bias/Regularizer/addAddV2%dense/bias/Regularizer/add/x:output:0dense/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0├
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp.dense_1_statefulpartitionedcall_dense_1_kernel ^dense_1/StatefulPartitionedCall*
dtype0* 
_output_shapes
:
ААР
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0* 
_output_shapes
:
АА*
T0q
 dense_1/kernel/Regularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:Ш
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/mul/xConst*
valueB
 *oГ:*
_output_shapes
: *
dtype0Ъ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: Ч
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ║
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp,dense_1_statefulpartitionedcall_dense_1_bias ^dense_1/StatefulPartitionedCall*
dtype0*
_output_shapes	
:АЗ
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0h
dense_1/bias/Regularizer/ConstConst*
valueB: *
dtype0*
_output_shapes
:Т
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/mul/xConst*
dtype0*
valueB
 *╜7Ж5*
_output_shapes
: Ф
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense_1/bias/Regularizer/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: С
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
_output_shapes
: *
T0Є
IdentityIdentity)tf_op_layer_Tanh/PartitionedCall:output:0^dense/StatefulPartitionedCall-^dense/bias/Regularizer/Square/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp ^dense_1/StatefulPartitionedCall/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCallW^tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::2\
,dense/bias/Regularizer/Square/ReadVariableOp,dense/bias/Regularizer/Square/ReadVariableOp2░
Vtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCallVtf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal/StatefulPartitionedCall2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall:. *
(
_user_specified_namepolicy_input_0: : : : : : : : :	 :
 : : 
а
k
M__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_fn_73376
inputs_0
identity╦
PartitionedCallPartitionedCallinputs_0*-
config_proto

GPU

CPU2*0J 8*q
flRj
h__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_and_return_conditional_losses_72061*
Tout
2*0
_output_shapes
:                  *,
_gradient_op_typePartitionedCall-72068*
Tin
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*/
_input_shapes
:                  :( $
"
_user_specified_name
inputs/0
╖
ж
__inference_loss_fn_2_73596C
?dense_1_kernel_regularizer_square_readvariableop_dense_1_kernel
identityИв0dense_1/kernel/Regularizer/Square/ReadVariableOp▓
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?dense_1_kernel_regularizer_square_readvariableop_dense_1_kernel*
dtype0* 
_output_shapes
:
ААР
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ААq
 dense_1/kernel/Regularizer/ConstConst*
dtype0*
valueB"       *
_output_shapes
:Ш
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/mul/xConst*
valueB
 *oГ:*
_output_shapes
: *
dtype0Ъ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0Ч
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0М
IdentityIdentity"dense_1/kernel/Regularizer/add:z:01^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
:2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:  
ї
╢
'__inference_dense_1_layer_call_fn_73221

inputs*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputs&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_71778*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         А*
Tout
2*,
_gradient_op_typePartitionedCall-71785Г
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
ю
f
J__inference_tf_op_layer_Exp_layer_call_and_return_conditional_losses_72324

inputs
identityD
ExpExpinputs*
T0*'
_output_shapes
:         O
IdentityIdentityExp:y:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
╣
|
^__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_and_return_conditional_losses_73466
inputs_0
identity^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>В
clip_by_value_1/MinimumMinimuminputs_0"clip_by_value_1/Minimum/y:output:0*'
_output_shapes
:         *
T0c
IdentityIdentityclip_by_value_1/Minimum:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
│
z
^__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_and_return_conditional_losses_72194

inputs
identity^
clip_by_value_1/Minimum/yConst*
valueB
 *ЪЩЩ>*
dtype0*
_output_shapes
: А
clip_by_value_1/MinimumMinimuminputs"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:         c
IdentityIdentityclip_by_value_1/Minimum:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
╚
╔
B__inference_dense_1_layer_call_and_return_conditional_losses_73214

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв.dense_1/bias/Regularizer/Square/ReadVariableOpв0dense_1/kernel/Regularizer/Square/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0w
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
dtype0*
_output_shapes	
:Аw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ап
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel^MatMul/ReadVariableOp*
dtype0* 
_output_shapes
:
ААР
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0* 
_output_shapes
:
АА*
T0q
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ш
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0e
 dense_1/kernel/Regularizer/mul/xConst*
dtype0*
valueB
 *oГ:*
_output_shapes
: Ъ
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
 dense_1/kernel/Regularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: Ч
dense_1/kernel/Regularizer/addAddV2)dense_1/kernel/Regularizer/add/x:output:0"dense_1/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0и
.dense_1/bias/Regularizer/Square/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias^BiasAdd/ReadVariableOp*
_output_shapes	
:А*
dtype0З
dense_1/bias/Regularizer/SquareSquare6dense_1/bias/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes	
:А*
T0h
dense_1/bias/Regularizer/ConstConst*
valueB: *
_output_shapes
:*
dtype0Т
dense_1/bias/Regularizer/SumSum#dense_1/bias/Regularizer/Square:y:0'dense_1/bias/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *╜7Ж5Ф
dense_1/bias/Regularizer/mulMul'dense_1/bias/Regularizer/mul/x:output:0%dense_1/bias/Regularizer/Sum:output:0*
_output_shapes
: *
T0c
dense_1/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    С
dense_1/bias/Regularizer/addAddV2'dense_1/bias/Regularizer/add/x:output:0 dense_1/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: ю
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_1/bias/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2`
.dense_1/bias/Regularizer/Square/ReadVariableOp.dense_1/bias/Regularizer/Square/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
∙
a
E__inference_activation_layer_call_and_return_conditional_losses_71737

inputs
identityG
ReluReluinputs*(
_output_shapes
:         А*
T0[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
░
{
]__inference_tf_op_layer_Normal_1/sample/concat_layer_call_and_return_conditional_losses_73350
inputs_0
identityg
Normal_1/sample/BroadcastArgsConst*
dtype0*
valueB:*
_output_shapes
:]
Normal_1/sample/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: и
Normal_1/sample/concatConcatV2inputs_0&Normal_1/sample/BroadcastArgs:output:0$Normal_1/sample/concat/axis:output:0*
_output_shapes
:*
T0*
NZ
IdentityIdentityNormal_1/sample/concat:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*
_input_shapes
::( $
"
_user_specified_name
inputs/0
ё
a
C__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_fn_73471
inputs_0
identity╕
PartitionedCallPartitionedCallinputs_0*,
_gradient_op_typePartitionedCall-72201*g
fbR`
^__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_and_return_conditional_losses_72194*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         *
Tout
2`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
╥
v
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_73523
inputs_0
inputs_1
identityP
mulMulinputs_0inputs_1*
T0*'
_output_shapes
:         O
IdentityIdentitymul:z:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*B
_input_shapes1
/:                  :         :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
к
y
]__inference_tf_op_layer_Normal_1/sample/concat_layer_call_and_return_conditional_losses_72020

inputs
identityg
Normal_1/sample/BroadcastArgsConst*
dtype0*
valueB:*
_output_shapes
:]
Normal_1/sample/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: ж
Normal_1/sample/concatConcatV2inputs&Normal_1/sample/BroadcastArgs:output:0$Normal_1/sample/concat/axis:output:0*
T0*
N*
_output_shapes
:Z
IdentityIdentityNormal_1/sample/concat:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*
_input_shapes
::& "
 
_user_specified_nameinputs
Е
а
__inference_loss_fn_0_73564?
;dense_kernel_regularizer_square_readvariableop_dense_kernel
identityИв.dense/kernel/Regularizer/Square/ReadVariableOpл
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_kernel_regularizer_square_readvariableop_dense_kernel*
_output_shapes
:	А*
dtype0Л
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
_output_shapes
:	А*
T0o
dense/kernel/Regularizer/ConstConst*
dtype0*
_output_shapes
:*
valueB"       Т
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
_output_shapes
: *
T0c
dense/kernel/Regularizer/mul/xConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: Ф
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: c
dense/kernel/Regularizer/add/xConst*
valueB
 *    *
_output_shapes
: *
dtype0С
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
_output_shapes
: *
T0И
IdentityIdentity dense/kernel/Regularizer/add:z:0/^dense/kernel/Regularizer/Square/ReadVariableOp*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
:2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:  
Є#
╫
__inference__traced_save_73676
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f80028e70bd943d3b2ff5ef8bea6e265/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ■
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*з
valueЭBЪB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0Е
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*+
value"B B B B B B B B B B B B B *
_output_shapes
:╙
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop"/device:CPU:0*
dtypes
2*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueB
B *
_output_shapes
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
_output_shapes
:*
NЦ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*z
_input_shapesi
g: :	А:А:А:А:
АА:А:А:А:	А::	А:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : 
Ї
h
J__inference_tf_op_layer_Exp_layer_call_and_return_conditional_losses_73512
inputs_0
identityF
ExpExpinputs_0*
T0*'
_output_shapes
:         O
IdentityIdentityExp:y:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
с
Y
;__inference_tf_op_layer_clip_by_value_1_layer_call_fn_73495
inputs_0
identity░
PartitionedCallPartitionedCallinputs_0*
Tout
2*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*,
_gradient_op_typePartitionedCall-72266*_
fZRX
V__inference_tf_op_layer_clip_by_value_1_layer_call_and_return_conditional_losses_72259*
Tin
2`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :( $
"
_user_specified_name
inputs/0
∙
a
E__inference_activation_layer_call_and_return_conditional_losses_73167

inputs
identityG
ReluReluinputs*(
_output_shapes
:         А*
T0[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╠
o
C__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_fn_73507
inputs_0
inputs_1
identity╠
PartitionedCallPartitionedCallinputs_0inputs_1*,
_gradient_op_typePartitionedCall-72311*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:                  *g
fbR`
^__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_and_return_conditional_losses_72303*
Tout
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*,
_input_shapes
:         ::( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
Ї
y
[__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_and_return_conditional_losses_73328
inputs_0
identityX
Normal_1/sample/ConstConst*
_output_shapes
: *
valueB *
dtype0g
Normal_1/sample/ProdProdinputs_0Normal_1/sample/Const:output:0*
T0*
_output_shapes
: T
IdentityIdentityNormal_1/sample/Prod:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
: :( $
"
_user_specified_name
inputs/0
╟
▀
5__inference_layer_normalization_1_layer_call_fn_73250

inputs7
3statefulpartitionedcall_layer_normalization_1_gamma6
2statefulpartitionedcall_layer_normalization_1_beta
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputs3statefulpartitionedcall_layer_normalization_1_gamma2statefulpartitionedcall_layer_normalization_1_beta*(
_output_shapes
:         А*
Tin
2*Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_71819*,
_gradient_op_typePartitionedCall-71826*
Tout
2*-
config_proto

GPU

CPU2*0J 8Г
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
т
Ц
#__inference_signature_wrapper_72767
policy_input_0(
$statefulpartitionedcall_dense_kernel&
"statefulpartitionedcall_dense_bias5
1statefulpartitionedcall_layer_normalization_gamma4
0statefulpartitionedcall_layer_normalization_beta*
&statefulpartitionedcall_dense_1_kernel(
$statefulpartitionedcall_dense_1_bias7
3statefulpartitionedcall_layer_normalization_1_gamma6
2statefulpartitionedcall_layer_normalization_1_beta*
&statefulpartitionedcall_dense_2_kernel(
$statefulpartitionedcall_dense_2_bias*
&statefulpartitionedcall_dense_3_kernel(
$statefulpartitionedcall_dense_3_bias
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallpolicy_input_0$statefulpartitionedcall_dense_kernel"statefulpartitionedcall_dense_bias1statefulpartitionedcall_layer_normalization_gamma0statefulpartitionedcall_layer_normalization_beta&statefulpartitionedcall_dense_1_kernel$statefulpartitionedcall_dense_1_bias3statefulpartitionedcall_layer_normalization_1_gamma2statefulpartitionedcall_layer_normalization_1_beta&statefulpartitionedcall_dense_2_kernel$statefulpartitionedcall_dense_2_bias&statefulpartitionedcall_dense_3_kernel$statefulpartitionedcall_dense_3_bias*
Tin
2*
Tout
2*'
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*)
f$R"
 __inference__wrapped_model_71639*,
_gradient_op_typePartitionedCall-72752В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*V
_input_shapesE
C:         ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_namepolicy_input_0: : : : : : : : :	 :
 : : "ЖL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*┴
serving_defaultн
I
policy_input_07
 serving_default_policy_input_0:0         D
tf_op_layer_Tanh0
StatefulPartitionedCall:0         tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:т╡
лю
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer_with_weights-5
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%
signatures
ю_default_save_signature
+я&call_and_return_all_conditional_losses
Ё__call__"Щш
_tf_keras_model■ч{"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": null, "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"name": "policy_input_0", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 28], "dtype": "float32", "sparse": false, "ragged": false, "name": "policy_input_0"}, "inbound_nodes": []}, {"name": "dense", "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999974752427e-07}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["policy_input_0", 0, 0, {}]]]}, {"name": "layer_normalization", "class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "gamma_initializer": {"class_name": "Ones", "config": {"dtype": "float32"}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"name": "activation", "class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}, {"name": "dense_1", "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999974752427e-07}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"name": "layer_normalization_1", "class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "gamma_initializer": {"class_name": "Ones", "config": {"dtype": "float32"}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"name": "activation_1", "class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]]}, {"name": "dense_2", "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"name": "tf_op_layer_clip_by_value/Minimum", "class_name": "TensorFlowOpLayer", "config": {"name": "clip_by_value/Minimum", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value/Minimum", "op": "Minimum", "input": ["dense_2/BiasAdd", "clip_by_value/Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2.0}}, "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"name": "tf_op_layer_clip_by_value", "class_name": "TensorFlowOpLayer", "config": {"name": "clip_by_value", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value", "op": "Maximum", "input": ["clip_by_value/Minimum", "clip_by_value/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -2.0}}, "inbound_nodes": [[["tf_op_layer_clip_by_value/Minimum", 0, 0, {}]]]}, {"name": "tf_op_layer_Shape", "class_name": "TensorFlowOpLayer", "config": {"name": "Shape", "trainable": true, "dtype": null, "node_def": {"name": "Shape", "op": "Shape", "input": ["clip_by_value"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT32"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_clip_by_value", 0, 0, {}]]]}, {"name": "tf_op_layer_strided_slice", "class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice", "trainable": true, "dtype": null, "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["Shape", "strided_slice/stack", "strided_slice/stack_1", "strided_slice/stack_2"], "attr": {"shrink_axis_mask": {"i": "1"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "0"}, "T": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "inbound_nodes": [[["tf_op_layer_Shape", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/Prod", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/Prod", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/Prod", "op": "Prod", "input": ["strided_slice", "Normal_1/sample/Const"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}, "T": {"type": "DT_INT32"}}}, "constants": {"1": []}}, "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/concat/values_0", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/concat/values_0", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/concat/values_0", "op": "Pack", "input": ["Normal_1/sample/Prod"], "attr": {"N": {"i": "1"}, "T": {"type": "DT_INT32"}, "axis": {"i": "0"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/Prod", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/concat", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/concat", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/concat", "op": "ConcatV2", "input": ["Normal_1/sample/concat/values_0", "Normal_1/sample/BroadcastArgs", "Normal_1/sample/concat/axis"], "attr": {"T": {"type": "DT_INT32"}, "N": {"i": "2"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": [8], "2": 0}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/concat/values_0", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/random_normal/RandomStandardNormal", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/random_normal/RandomStandardNormal", "op": "RandomStandardNormal", "input": ["Normal_1/sample/concat"], "attr": {"dtype": {"type": "DT_FLOAT"}, "T": {"type": "DT_INT32"}, "seed": {"i": "0"}, "seed2": {"i": "0"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/concat", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/random_normal/mul", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/random_normal/mul", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/random_normal/mul", "op": "Mul", "input": ["Normal_1/sample/random_normal/RandomStandardNormal", "Normal_1/sample/random_normal/stddev"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/random_normal", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/random_normal", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/random_normal", "op": "Add", "input": ["Normal_1/sample/random_normal/mul", "Normal_1/sample/random_normal/mean"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.0}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/random_normal/mul", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/mul", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/mul", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/mul", "op": "Mul", "input": ["Normal_1/sample/random_normal", "ones"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/random_normal", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/add", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/add", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/add", "op": "AddV2", "input": ["Normal_1/sample/mul", "zeros"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/mul", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/Shape_2", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/Shape_2", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/Shape_2", "op": "Shape", "input": ["Normal_1/sample/add"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT32"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/add", 0, 0, {}]]]}, {"name": "dense_3", "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/expand_to_vector/Reshape", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/expand_to_vector/Reshape", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/expand_to_vector/Reshape", "op": "Reshape", "input": ["strided_slice", "Normal_1/sample/expand_to_vector/Reshape/shape"], "attr": {"Tshape": {"type": "DT_INT32"}, "T": {"type": "DT_INT32"}}}, "constants": {"1": [1]}}, "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/strided_slice", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/strided_slice", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/strided_slice", "op": "StridedSlice", "input": ["Normal_1/sample/Shape_2", "Normal_1/sample/strided_slice/stack", "Normal_1/sample/strided_slice/stack_1", "Normal_1/sample/strided_slice/stack_2"], "attr": {"ellipsis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_INT32"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/Shape_2", 0, 0, {}]]]}, {"name": "tf_op_layer_clip_by_value_1/Minimum", "class_name": "TensorFlowOpLayer", "config": {"name": "clip_by_value_1/Minimum", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value_1/Minimum", "op": "Minimum", "input": ["dense_3/BiasAdd", "clip_by_value_1/Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.30000001192092896}}, "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/concat_1", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/concat_1", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/concat_1", "op": "ConcatV2", "input": ["Normal_1/sample/expand_to_vector/Reshape", "Normal_1/sample/strided_slice", "Normal_1/sample/concat_1/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "N": {"i": "2"}, "T": {"type": "DT_INT32"}}}, "constants": {"2": 0}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/expand_to_vector/Reshape", 0, 0, {}], ["tf_op_layer_Normal_1/sample/strided_slice", 0, 0, {}]]]}, {"name": "tf_op_layer_clip_by_value_1", "class_name": "TensorFlowOpLayer", "config": {"name": "clip_by_value_1", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value_1", "op": "Maximum", "input": ["clip_by_value_1/Minimum", "clip_by_value_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -20.0}}, "inbound_nodes": [[["tf_op_layer_clip_by_value_1/Minimum", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/Reshape", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/Reshape", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/Reshape", "op": "Reshape", "input": ["Normal_1/sample/add", "Normal_1/sample/concat_1"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/add", 0, 0, {}], ["tf_op_layer_Normal_1/sample/concat_1", 0, 0, {}]]]}, {"name": "tf_op_layer_Exp", "class_name": "TensorFlowOpLayer", "config": {"name": "Exp", "trainable": true, "dtype": null, "node_def": {"name": "Exp", "op": "Exp", "input": ["clip_by_value_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_clip_by_value_1", 0, 0, {}]]]}, {"name": "tf_op_layer_mul", "class_name": "TensorFlowOpLayer", "config": {"name": "mul", "trainable": true, "dtype": null, "node_def": {"name": "mul", "op": "Mul", "input": ["Normal_1/sample/Reshape", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/Reshape", 0, 0, {}], ["tf_op_layer_Exp", 0, 0, {}]]]}, {"name": "tf_op_layer_add", "class_name": "TensorFlowOpLayer", "config": {"name": "add", "trainable": true, "dtype": null, "node_def": {"name": "add", "op": "AddV2", "input": ["clip_by_value", "mul"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_clip_by_value", 0, 0, {}], ["tf_op_layer_mul", 0, 0, {}]]]}, {"name": "tf_op_layer_Tanh", "class_name": "TensorFlowOpLayer", "config": {"name": "Tanh", "trainable": true, "dtype": null, "node_def": {"name": "Tanh", "op": "Tanh", "input": ["add"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_add", 0, 0, {}]]]}], "input_layers": [["policy_input_0", 0, 0]], "output_layers": [["tf_op_layer_Tanh", 0, 0]]}, "input_spec": null, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"name": "policy_input_0", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 28], "dtype": "float32", "sparse": false, "ragged": false, "name": "policy_input_0"}, "inbound_nodes": []}, {"name": "dense", "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999974752427e-07}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["policy_input_0", 0, 0, {}]]]}, {"name": "layer_normalization", "class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "gamma_initializer": {"class_name": "Ones", "config": {"dtype": "float32"}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"name": "activation", "class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["layer_normalization", 0, 0, {}]]]}, {"name": "dense_1", "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999974752427e-07}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"name": "layer_normalization_1", "class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "gamma_initializer": {"class_name": "Ones", "config": {"dtype": "float32"}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"name": "activation_1", "class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]]}, {"name": "dense_2", "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"name": "tf_op_layer_clip_by_value/Minimum", "class_name": "TensorFlowOpLayer", "config": {"name": "clip_by_value/Minimum", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value/Minimum", "op": "Minimum", "input": ["dense_2/BiasAdd", "clip_by_value/Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2.0}}, "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"name": "tf_op_layer_clip_by_value", "class_name": "TensorFlowOpLayer", "config": {"name": "clip_by_value", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value", "op": "Maximum", "input": ["clip_by_value/Minimum", "clip_by_value/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -2.0}}, "inbound_nodes": [[["tf_op_layer_clip_by_value/Minimum", 0, 0, {}]]]}, {"name": "tf_op_layer_Shape", "class_name": "TensorFlowOpLayer", "config": {"name": "Shape", "trainable": true, "dtype": null, "node_def": {"name": "Shape", "op": "Shape", "input": ["clip_by_value"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT32"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_clip_by_value", 0, 0, {}]]]}, {"name": "tf_op_layer_strided_slice", "class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice", "trainable": true, "dtype": null, "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["Shape", "strided_slice/stack", "strided_slice/stack_1", "strided_slice/stack_2"], "attr": {"shrink_axis_mask": {"i": "1"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "0"}, "T": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "inbound_nodes": [[["tf_op_layer_Shape", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/Prod", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/Prod", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/Prod", "op": "Prod", "input": ["strided_slice", "Normal_1/sample/Const"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}, "T": {"type": "DT_INT32"}}}, "constants": {"1": []}}, "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/concat/values_0", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/concat/values_0", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/concat/values_0", "op": "Pack", "input": ["Normal_1/sample/Prod"], "attr": {"N": {"i": "1"}, "T": {"type": "DT_INT32"}, "axis": {"i": "0"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/Prod", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/concat", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/concat", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/concat", "op": "ConcatV2", "input": ["Normal_1/sample/concat/values_0", "Normal_1/sample/BroadcastArgs", "Normal_1/sample/concat/axis"], "attr": {"T": {"type": "DT_INT32"}, "N": {"i": "2"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": [8], "2": 0}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/concat/values_0", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/random_normal/RandomStandardNormal", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/random_normal/RandomStandardNormal", "op": "RandomStandardNormal", "input": ["Normal_1/sample/concat"], "attr": {"dtype": {"type": "DT_FLOAT"}, "T": {"type": "DT_INT32"}, "seed": {"i": "0"}, "seed2": {"i": "0"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/concat", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/random_normal/mul", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/random_normal/mul", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/random_normal/mul", "op": "Mul", "input": ["Normal_1/sample/random_normal/RandomStandardNormal", "Normal_1/sample/random_normal/stddev"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/random_normal", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/random_normal", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/random_normal", "op": "Add", "input": ["Normal_1/sample/random_normal/mul", "Normal_1/sample/random_normal/mean"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.0}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/random_normal/mul", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/mul", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/mul", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/mul", "op": "Mul", "input": ["Normal_1/sample/random_normal", "ones"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/random_normal", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/add", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/add", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/add", "op": "AddV2", "input": ["Normal_1/sample/mul", "zeros"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/mul", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/Shape_2", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/Shape_2", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/Shape_2", "op": "Shape", "input": ["Normal_1/sample/add"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT32"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/add", 0, 0, {}]]]}, {"name": "dense_3", "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/expand_to_vector/Reshape", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/expand_to_vector/Reshape", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/expand_to_vector/Reshape", "op": "Reshape", "input": ["strided_slice", "Normal_1/sample/expand_to_vector/Reshape/shape"], "attr": {"Tshape": {"type": "DT_INT32"}, "T": {"type": "DT_INT32"}}}, "constants": {"1": [1]}}, "inbound_nodes": [[["tf_op_layer_strided_slice", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/strided_slice", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/strided_slice", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/strided_slice", "op": "StridedSlice", "input": ["Normal_1/sample/Shape_2", "Normal_1/sample/strided_slice/stack", "Normal_1/sample/strided_slice/stack_1", "Normal_1/sample/strided_slice/stack_2"], "attr": {"ellipsis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_INT32"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/Shape_2", 0, 0, {}]]]}, {"name": "tf_op_layer_clip_by_value_1/Minimum", "class_name": "TensorFlowOpLayer", "config": {"name": "clip_by_value_1/Minimum", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value_1/Minimum", "op": "Minimum", "input": ["dense_3/BiasAdd", "clip_by_value_1/Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.30000001192092896}}, "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/concat_1", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/concat_1", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/concat_1", "op": "ConcatV2", "input": ["Normal_1/sample/expand_to_vector/Reshape", "Normal_1/sample/strided_slice", "Normal_1/sample/concat_1/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "N": {"i": "2"}, "T": {"type": "DT_INT32"}}}, "constants": {"2": 0}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/expand_to_vector/Reshape", 0, 0, {}], ["tf_op_layer_Normal_1/sample/strided_slice", 0, 0, {}]]]}, {"name": "tf_op_layer_clip_by_value_1", "class_name": "TensorFlowOpLayer", "config": {"name": "clip_by_value_1", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value_1", "op": "Maximum", "input": ["clip_by_value_1/Minimum", "clip_by_value_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -20.0}}, "inbound_nodes": [[["tf_op_layer_clip_by_value_1/Minimum", 0, 0, {}]]]}, {"name": "tf_op_layer_Normal_1/sample/Reshape", "class_name": "TensorFlowOpLayer", "config": {"name": "Normal_1/sample/Reshape", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/Reshape", "op": "Reshape", "input": ["Normal_1/sample/add", "Normal_1/sample/concat_1"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/add", 0, 0, {}], ["tf_op_layer_Normal_1/sample/concat_1", 0, 0, {}]]]}, {"name": "tf_op_layer_Exp", "class_name": "TensorFlowOpLayer", "config": {"name": "Exp", "trainable": true, "dtype": null, "node_def": {"name": "Exp", "op": "Exp", "input": ["clip_by_value_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_clip_by_value_1", 0, 0, {}]]]}, {"name": "tf_op_layer_mul", "class_name": "TensorFlowOpLayer", "config": {"name": "mul", "trainable": true, "dtype": null, "node_def": {"name": "mul", "op": "Mul", "input": ["Normal_1/sample/Reshape", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_Normal_1/sample/Reshape", 0, 0, {}], ["tf_op_layer_Exp", 0, 0, {}]]]}, {"name": "tf_op_layer_add", "class_name": "TensorFlowOpLayer", "config": {"name": "add", "trainable": true, "dtype": null, "node_def": {"name": "add", "op": "AddV2", "input": ["clip_by_value", "mul"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_clip_by_value", 0, 0, {}], ["tf_op_layer_mul", 0, 0, {}]]]}, {"name": "tf_op_layer_Tanh", "class_name": "TensorFlowOpLayer", "config": {"name": "Tanh", "trainable": true, "dtype": null, "node_def": {"name": "Tanh", "op": "Tanh", "input": ["add"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "inbound_nodes": [[["tf_op_layer_add", 0, 0, {}]]]}], "input_layers": [["policy_input_0", 0, 0]], "output_layers": [["tf_op_layer_Tanh", 0, 0]]}}}
Ў
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+ё&call_and_return_all_conditional_losses
Є__call__"х
_tf_keras_layer╦{"class_name": "InputLayer", "name": "policy_input_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28], "config": {"batch_input_shape": [null, 28], "dtype": "float32", "sparse": false, "ragged": false, "name": "policy_input_0"}, "input_spec": null, "activity_regularizer": null}
╪

*kernel
+_callable_losses
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"Ы
_tf_keras_layerБ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999974752427e-07}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 28}}}, "activity_regularizer": null}
╚
1axis
	2gamma
3beta
4_callable_losses
5	variables
6trainable_variables
7regularization_losses
8	keras_api
+ї&call_and_return_all_conditional_losses
Ў__call__"В
_tf_keras_layerш{"class_name": "LayerNormalization", "name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "gamma_initializer": {"class_name": "Ones", "config": {"dtype": "float32"}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": null, "activity_regularizer": null}
х
9_callable_losses
:	variables
;trainable_variables
<regularization_losses
=	keras_api
+ў&call_and_return_all_conditional_losses
°__call__"╛
_tf_keras_layerд{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "input_spec": null, "activity_regularizer": null}
▌

>kernel
?_callable_losses
@bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
+∙&call_and_return_all_conditional_losses
·__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999974752427e-07}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "activity_regularizer": null}
╠
Eaxis
	Fgamma
Gbeta
H_callable_losses
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
+√&call_and_return_all_conditional_losses
№__call__"Ж
_tf_keras_layerь{"class_name": "LayerNormalization", "name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "gamma_initializer": {"class_name": "Ones", "config": {"dtype": "float32"}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": null, "activity_regularizer": null}
щ
M_callable_losses
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
+¤&call_and_return_all_conditional_losses
■__call__"┬
_tf_keras_layerи{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "input_spec": null, "activity_regularizer": null}
╧

Rkernel
Sbias
T_callable_losses
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
+ &call_and_return_all_conditional_losses
А__call__"Т
_tf_keras_layer°{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "activity_regularizer": null}
┤
Y	constants
Z_callable_losses
[	variables
\trainable_variables
]regularization_losses
^	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"■
_tf_keras_layerф{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_clip_by_value/Minimum", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "clip_by_value/Minimum", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value/Minimum", "op": "Minimum", "input": ["dense_2/BiasAdd", "clip_by_value/Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2.0}}, "input_spec": null, "activity_regularizer": null}
Ы
_	constants
`_callable_losses
a	variables
btrainable_variables
cregularization_losses
d	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"х
_tf_keras_layer╦{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_clip_by_value", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "clip_by_value", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value", "op": "Maximum", "input": ["clip_by_value/Minimum", "clip_by_value/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -2.0}}, "input_spec": null, "activity_regularizer": null}
 
e	constants
f_callable_losses
g	variables
htrainable_variables
iregularization_losses
j	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"╔
_tf_keras_layerп{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Shape", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Shape", "trainable": true, "dtype": null, "node_def": {"name": "Shape", "op": "Shape", "input": ["clip_by_value"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT32"}}}, "constants": {}}, "input_spec": null, "activity_regularizer": null}
Д
k	constants
l_callable_losses
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
+З&call_and_return_all_conditional_losses
И__call__"╬
_tf_keras_layer┤{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "strided_slice", "trainable": true, "dtype": null, "node_def": {"name": "strided_slice", "op": "StridedSlice", "input": ["Shape", "strided_slice/stack", "strided_slice/stack_1", "strided_slice/stack_2"], "attr": {"shrink_axis_mask": {"i": "1"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "0"}, "T": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "0"}}}, "constants": {"1": [0], "2": [1], "3": [1]}}, "input_spec": null, "activity_regularizer": null}
т
q	constants
r_callable_losses
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"м
_tf_keras_layerТ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/Prod", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/Prod", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/Prod", "op": "Prod", "input": ["strided_slice", "Normal_1/sample/Const"], "attr": {"Tidx": {"type": "DT_INT32"}, "keep_dims": {"b": false}, "T": {"type": "DT_INT32"}}}, "constants": {"1": []}}, "input_spec": null, "activity_regularizer": null}
╓
w	constants
x_callable_losses
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"а
_tf_keras_layerЖ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/concat/values_0", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/concat/values_0", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/concat/values_0", "op": "Pack", "input": ["Normal_1/sample/Prod"], "attr": {"N": {"i": "1"}, "T": {"type": "DT_INT32"}, "axis": {"i": "0"}}}, "constants": {}}, "input_spec": null, "activity_regularizer": null}
з
}	constants
~_callable_losses
	variables
Аtrainable_variables
Бregularization_losses
В	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"ю
_tf_keras_layer╘{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/concat", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/concat", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/concat", "op": "ConcatV2", "input": ["Normal_1/sample/concat/values_0", "Normal_1/sample/BroadcastArgs", "Normal_1/sample/concat/axis"], "attr": {"T": {"type": "DT_INT32"}, "N": {"i": "2"}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": [8], "2": 0}}, "input_spec": null, "activity_regularizer": null}
╩
Г	constants
Д_callable_losses
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"О
_tf_keras_layerЇ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/random_normal/RandomStandardNormal", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/random_normal/RandomStandardNormal", "op": "RandomStandardNormal", "input": ["Normal_1/sample/concat"], "attr": {"dtype": {"type": "DT_FLOAT"}, "T": {"type": "DT_INT32"}, "seed": {"i": "0"}, "seed2": {"i": "0"}}}, "constants": {}}, "input_spec": null, "activity_regularizer": null}
К
Й	constants
К_callable_losses
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"╬
_tf_keras_layer┤{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/random_normal/mul", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/random_normal/mul", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/random_normal/mul", "op": "Mul", "input": ["Normal_1/sample/random_normal/RandomStandardNormal", "Normal_1/sample/random_normal/stddev"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "input_spec": null, "activity_regularizer": null}
ы
П	constants
Р_callable_losses
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"п
_tf_keras_layerХ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/random_normal", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/random_normal", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/random_normal", "op": "Add", "input": ["Normal_1/sample/random_normal/mul", "Normal_1/sample/random_normal/mean"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.0}}, "input_spec": null, "activity_regularizer": null}
╨
Х	constants
Ц_callable_losses
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"Ф
_tf_keras_layer·{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/mul", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/mul", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/mul", "op": "Mul", "input": ["Normal_1/sample/random_normal", "ones"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}, "input_spec": null, "activity_regularizer": null}
╔
Ы	constants
Ь_callable_losses
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"Н
_tf_keras_layerє{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/add", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/add", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/add", "op": "AddV2", "input": ["Normal_1/sample/mul", "zeros"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}, "input_spec": null, "activity_regularizer": null}
┴
б	constants
в_callable_losses
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"Е
_tf_keras_layerы{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/Shape_2", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/Shape_2", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/Shape_2", "op": "Shape", "input": ["Normal_1/sample/add"], "attr": {"T": {"type": "DT_FLOAT"}, "out_type": {"type": "DT_INT32"}}}, "constants": {}}, "input_spec": null, "activity_regularizer": null}
╓
зkernel
	иbias
й_callable_losses
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"Т
_tf_keras_layer°{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {"dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "activity_regularizer": null}
и
о	constants
п_callable_losses
░	variables
▒trainable_variables
▓regularization_losses
│	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"ь
_tf_keras_layer╥{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/expand_to_vector/Reshape", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/expand_to_vector/Reshape", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/expand_to_vector/Reshape", "op": "Reshape", "input": ["strided_slice", "Normal_1/sample/expand_to_vector/Reshape/shape"], "attr": {"Tshape": {"type": "DT_INT32"}, "T": {"type": "DT_INT32"}}}, "constants": {"1": [1]}}, "input_spec": null, "activity_regularizer": null}
№
┤	constants
╡_callable_losses
╢	variables
╖trainable_variables
╕regularization_losses
╣	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"└
_tf_keras_layerж{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/strided_slice", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/strided_slice", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/strided_slice", "op": "StridedSlice", "input": ["Normal_1/sample/Shape_2", "Normal_1/sample/strided_slice/stack", "Normal_1/sample/strided_slice/stack_1", "Normal_1/sample/strided_slice/stack_2"], "attr": {"ellipsis_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_INT32"}}}, "constants": {"1": [1], "2": [0], "3": [1]}}, "input_spec": null, "activity_regularizer": null}
╥
║	constants
╗_callable_losses
╝	variables
╜trainable_variables
╛regularization_losses
┐	keras_api
+б&call_and_return_all_conditional_losses
в__call__"Ц
_tf_keras_layer№{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_clip_by_value_1/Minimum", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "clip_by_value_1/Minimum", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value_1/Minimum", "op": "Minimum", "input": ["dense_3/BiasAdd", "clip_by_value_1/Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.30000001192092896}}, "input_spec": null, "activity_regularizer": null}
▒
└	constants
┴_callable_losses
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
+г&call_and_return_all_conditional_losses
д__call__"ї
_tf_keras_layer█{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/concat_1", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/concat_1", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/concat_1", "op": "ConcatV2", "input": ["Normal_1/sample/expand_to_vector/Reshape", "Normal_1/sample/strided_slice", "Normal_1/sample/concat_1/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "N": {"i": "2"}, "T": {"type": "DT_INT32"}}}, "constants": {"2": 0}}, "input_spec": null, "activity_regularizer": null}
м
╞	constants
╟_callable_losses
╚	variables
╔trainable_variables
╩regularization_losses
╦	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"Ё
_tf_keras_layer╓{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_clip_by_value_1", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "clip_by_value_1", "trainable": true, "dtype": null, "node_def": {"name": "clip_by_value_1", "op": "Maximum", "input": ["clip_by_value_1/Minimum", "clip_by_value_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": -20.0}}, "input_spec": null, "activity_regularizer": null}
▌
╠	constants
═_callable_losses
╬	variables
╧trainable_variables
╨regularization_losses
╤	keras_api
+з&call_and_return_all_conditional_losses
и__call__"б
_tf_keras_layerЗ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Normal_1/sample/Reshape", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Normal_1/sample/Reshape", "trainable": true, "dtype": null, "node_def": {"name": "Normal_1/sample/Reshape", "op": "Reshape", "input": ["Normal_1/sample/add", "Normal_1/sample/concat_1"], "attr": {"T": {"type": "DT_FLOAT"}, "Tshape": {"type": "DT_INT32"}}}, "constants": {}}, "input_spec": null, "activity_regularizer": null}
▌
╥	constants
╙_callable_losses
╘	variables
╒trainable_variables
╓regularization_losses
╫	keras_api
+й&call_and_return_all_conditional_losses
к__call__"б
_tf_keras_layerЗ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Exp", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Exp", "trainable": true, "dtype": null, "node_def": {"name": "Exp", "op": "Exp", "input": ["clip_by_value_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "input_spec": null, "activity_regularizer": null}
ь
╪	constants
┘_callable_losses
┌	variables
█trainable_variables
▄regularization_losses
▌	keras_api
+л&call_and_return_all_conditional_losses
м__call__"░
_tf_keras_layerЦ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_mul", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "mul", "trainable": true, "dtype": null, "node_def": {"name": "mul", "op": "Mul", "input": ["Normal_1/sample/Reshape", "Exp"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "input_spec": null, "activity_regularizer": null}
ф
▐	constants
▀_callable_losses
р	variables
сtrainable_variables
тregularization_losses
у	keras_api
+н&call_and_return_all_conditional_losses
о__call__"и
_tf_keras_layerО{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_add", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "add", "trainable": true, "dtype": null, "node_def": {"name": "add", "op": "AddV2", "input": ["clip_by_value", "mul"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "input_spec": null, "activity_regularizer": null}
╒
ф	constants
х_callable_losses
ц	variables
чtrainable_variables
шregularization_losses
щ	keras_api
+п&call_and_return_all_conditional_losses
░__call__"Щ
_tf_keras_layer {"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Tanh", "trainable": true, "expects_training_arg": false, "dtype": null, "batch_input_shape": null, "config": {"name": "Tanh", "trainable": true, "dtype": null, "node_def": {"name": "Tanh", "op": "Tanh", "input": ["add"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "input_spec": null, "activity_regularizer": null}
x
*0
,1
22
33
>4
@5
F6
G7
R8
S9
з10
и11"
trackable_list_wrapper
x
*0
,1
22
33
>4
@5
F6
G7
R8
S9
з10
и11"
trackable_list_wrapper
@
▒0
▓1
│2
┤3"
trackable_list_wrapper
┐
!	variables
 ъlayer_regularization_losses
ыlayers
ьmetrics
"trainable_variables
#regularization_losses
эnon_trainable_variables
Ё__call__
ю_default_save_signature
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
-
╡serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
&	variables
 юlayer_regularization_losses
яlayers
Ёmetrics
'trainable_variables
(regularization_losses
ёnon_trainable_variables
Є__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
:	А2dense/kernel
 "
trackable_list_wrapper
:А2
dense/bias
.
*0
,1"
trackable_list_wrapper
.
*0
,1"
trackable_list_wrapper
0
▒0
▓1"
trackable_list_wrapper
б
-	variables
 Єlayer_regularization_losses
єlayers
Їmetrics
.trainable_variables
/regularization_losses
їnon_trainable_variables
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(:&А2layer_normalization/gamma
':%А2layer_normalization/beta
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
б
5	variables
 Ўlayer_regularization_losses
ўlayers
°metrics
6trainable_variables
7regularization_losses
∙non_trainable_variables
Ў__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
:	variables
 ·layer_regularization_losses
√layers
№metrics
;trainable_variables
<regularization_losses
¤non_trainable_variables
°__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
": 
АА2dense_1/kernel
 "
trackable_list_wrapper
:А2dense_1/bias
.
>0
@1"
trackable_list_wrapper
.
>0
@1"
trackable_list_wrapper
0
│0
┤1"
trackable_list_wrapper
б
A	variables
 ■layer_regularization_losses
 layers
Аmetrics
Btrainable_variables
Cregularization_losses
Бnon_trainable_variables
·__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2layer_normalization_1/gamma
):'А2layer_normalization_1/beta
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
б
I	variables
 Вlayer_regularization_losses
Гlayers
Дmetrics
Jtrainable_variables
Kregularization_losses
Еnon_trainable_variables
№__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
N	variables
 Жlayer_regularization_losses
Зlayers
Иmetrics
Otrainable_variables
Pregularization_losses
Йnon_trainable_variables
■__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
!:	А2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
б
U	variables
 Кlayer_regularization_losses
Лlayers
Мmetrics
Vtrainable_variables
Wregularization_losses
Нnon_trainable_variables
А__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
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
б
[	variables
 Оlayer_regularization_losses
Пlayers
Рmetrics
\trainable_variables
]regularization_losses
Сnon_trainable_variables
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
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
б
a	variables
 Тlayer_regularization_losses
Уlayers
Фmetrics
btrainable_variables
cregularization_losses
Хnon_trainable_variables
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
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
б
g	variables
 Цlayer_regularization_losses
Чlayers
Шmetrics
htrainable_variables
iregularization_losses
Щnon_trainable_variables
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
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
б
m	variables
 Ъlayer_regularization_losses
Ыlayers
Ьmetrics
ntrainable_variables
oregularization_losses
Эnon_trainable_variables
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
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
б
s	variables
 Юlayer_regularization_losses
Яlayers
аmetrics
ttrainable_variables
uregularization_losses
бnon_trainable_variables
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
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
б
y	variables
 вlayer_regularization_losses
гlayers
дmetrics
ztrainable_variables
{regularization_losses
еnon_trainable_variables
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
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
г
	variables
 жlayer_regularization_losses
зlayers
иmetrics
Аtrainable_variables
Бregularization_losses
йnon_trainable_variables
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
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
д
Е	variables
 кlayer_regularization_losses
лlayers
мmetrics
Жtrainable_variables
Зregularization_losses
нnon_trainable_variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
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
д
Л	variables
 оlayer_regularization_losses
пlayers
░metrics
Мtrainable_variables
Нregularization_losses
▒non_trainable_variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
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
д
С	variables
 ▓layer_regularization_losses
│layers
┤metrics
Тtrainable_variables
Уregularization_losses
╡non_trainable_variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
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
д
Ч	variables
 ╢layer_regularization_losses
╖layers
╕metrics
Шtrainable_variables
Щregularization_losses
╣non_trainable_variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
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
д
Э	variables
 ║layer_regularization_losses
╗layers
╝metrics
Юtrainable_variables
Яregularization_losses
╜non_trainable_variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
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
д
г	variables
 ╛layer_regularization_losses
┐layers
└metrics
дtrainable_variables
еregularization_losses
┴non_trainable_variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
!:	А2dense_3/kernel
:2dense_3/bias
 "
trackable_list_wrapper
0
з0
и1"
trackable_list_wrapper
0
з0
и1"
trackable_list_wrapper
 "
trackable_list_wrapper
д
к	variables
 ┬layer_regularization_losses
├layers
─metrics
лtrainable_variables
мregularization_losses
┼non_trainable_variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
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
д
░	variables
 ╞layer_regularization_losses
╟layers
╚metrics
▒trainable_variables
▓regularization_losses
╔non_trainable_variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
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
д
╢	variables
 ╩layer_regularization_losses
╦layers
╠metrics
╖trainable_variables
╕regularization_losses
═non_trainable_variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
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
д
╝	variables
 ╬layer_regularization_losses
╧layers
╨metrics
╜trainable_variables
╛regularization_losses
╤non_trainable_variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
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
д
┬	variables
 ╥layer_regularization_losses
╙layers
╘metrics
├trainable_variables
─regularization_losses
╒non_trainable_variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
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
д
╚	variables
 ╓layer_regularization_losses
╫layers
╪metrics
╔trainable_variables
╩regularization_losses
┘non_trainable_variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
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
д
╬	variables
 ┌layer_regularization_losses
█layers
▄metrics
╧trainable_variables
╨regularization_losses
▌non_trainable_variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
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
д
╘	variables
 ▐layer_regularization_losses
▀layers
рmetrics
╒trainable_variables
╓regularization_losses
сnon_trainable_variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
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
д
┌	variables
 тlayer_regularization_losses
уlayers
фmetrics
█trainable_variables
▄regularization_losses
хnon_trainable_variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
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
д
р	variables
 цlayer_regularization_losses
чlayers
шmetrics
сtrainable_variables
тregularization_losses
щnon_trainable_variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
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
д
ц	variables
 ъlayer_regularization_losses
ыlayers
ьmetrics
чtrainable_variables
шregularization_losses
эnon_trainable_variables
░__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
▒0
▓1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
│0
┤1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х2т
 __inference__wrapped_model_71639╜
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *-в*
(К%
policy_input_0         
╓2╙
B__inference_model_1_layer_call_and_return_conditional_losses_72910
B__inference_model_1_layer_call_and_return_conditional_losses_73050
B__inference_model_1_layer_call_and_return_conditional_losses_72436
B__inference_model_1_layer_call_and_return_conditional_losses_72518└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
'__inference_model_1_layer_call_fn_72716
'__inference_model_1_layer_call_fn_73067
'__inference_model_1_layer_call_fn_73084
'__inference_model_1_layer_call_fn_72616└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ъ2ч
@__inference_dense_layer_call_and_return_conditional_losses_73126в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
╧2╠
%__inference_dense_layer_call_fn_73133в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
°2ї
N__inference_layer_normalization_layer_call_and_return_conditional_losses_73155в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
▌2┌
3__inference_layer_normalization_layer_call_fn_73162в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
я2ь
E__inference_activation_layer_call_and_return_conditional_losses_73167в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
╘2╤
*__inference_activation_layer_call_fn_73172в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ь2щ
B__inference_dense_1_layer_call_and_return_conditional_losses_73214в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
╤2╬
'__inference_dense_1_layer_call_fn_73221в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
·2ў
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_73243в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
▀2▄
5__inference_layer_normalization_1_layer_call_fn_73250в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ё2ю
G__inference_activation_1_layer_call_and_return_conditional_losses_73255в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
╓2╙
,__inference_activation_1_layer_call_fn_73260в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ь2щ
B__inference_dense_2_layer_call_and_return_conditional_losses_73270в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
╤2╬
'__inference_dense_2_layer_call_fn_73277в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Ж2Г
\__inference_tf_op_layer_clip_by_value/Minimum_layer_call_and_return_conditional_losses_73283в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ы2ш
A__inference_tf_op_layer_clip_by_value/Minimum_layer_call_fn_73288в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
■2√
T__inference_tf_op_layer_clip_by_value_layer_call_and_return_conditional_losses_73294в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
у2р
9__inference_tf_op_layer_clip_by_value_layer_call_fn_73299в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Ў2є
L__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_73304в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
█2╪
1__inference_tf_op_layer_Shape_layer_call_fn_73309в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
■2√
T__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_73317в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
у2р
9__inference_tf_op_layer_strided_slice_layer_call_fn_73322в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Е2В
[__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_and_return_conditional_losses_73328в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ъ2ч
@__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_fn_73333в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Р2Н
f__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_and_return_conditional_losses_73338в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ї2Є
K__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_fn_73343в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
З2Д
]__inference_tf_op_layer_Normal_1/sample/concat_layer_call_and_return_conditional_losses_73350в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ь2щ
B__inference_tf_op_layer_Normal_1/sample/concat_layer_call_fn_73355в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
г2а
y__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_and_return_conditional_losses_73360в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
И2Е
^__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_fn_73365в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Т2П
h__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_and_return_conditional_losses_73371в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ў2Ї
M__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_fn_73376в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
О2Л
d__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_and_return_conditional_losses_73382в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
є2Ё
I__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_fn_73387в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Д2Б
Z__inference_tf_op_layer_Normal_1/sample/mul_layer_call_and_return_conditional_losses_73393в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
щ2ц
?__inference_tf_op_layer_Normal_1/sample/mul_layer_call_fn_73398в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Д2Б
Z__inference_tf_op_layer_Normal_1/sample/add_layer_call_and_return_conditional_losses_73404в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
щ2ц
?__inference_tf_op_layer_Normal_1/sample/add_layer_call_fn_73409в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
И2Е
^__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_and_return_conditional_losses_73414в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
э2ъ
C__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_fn_73419в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ь2щ
B__inference_dense_3_layer_call_and_return_conditional_losses_73429в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
╤2╬
'__inference_dense_3_layer_call_fn_73436в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Щ2Ц
o__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_and_return_conditional_losses_73442в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
■2√
T__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_fn_73447в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
О2Л
d__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_and_return_conditional_losses_73455в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
є2Ё
I__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_fn_73460в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
И2Е
^__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_and_return_conditional_losses_73466в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
э2ъ
C__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_fn_73471в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Й2Ж
___inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_and_return_conditional_losses_73478в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ю2ы
D__inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_fn_73484в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
А2¤
V__inference_tf_op_layer_clip_by_value_1_layer_call_and_return_conditional_losses_73490в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
х2т
;__inference_tf_op_layer_clip_by_value_1_layer_call_fn_73495в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
И2Е
^__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_and_return_conditional_losses_73501в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
э2ъ
C__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_fn_73507в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Ї2ё
J__inference_tf_op_layer_Exp_layer_call_and_return_conditional_losses_73512в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
┘2╓
/__inference_tf_op_layer_Exp_layer_call_fn_73517в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Ї2ё
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_73523в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
┘2╓
/__inference_tf_op_layer_mul_layer_call_fn_73529в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
Ї2ё
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_73535в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
┘2╓
/__inference_tf_op_layer_add_layer_call_fn_73541в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
ї2Є
K__inference_tf_op_layer_Tanh_layer_call_and_return_conditional_losses_73546в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
┌2╫
0__inference_tf_op_layer_Tanh_layer_call_fn_73551в
Щ▓Х
FullArgSpec
argsЪ
jself
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
annotationsк *
 
▓2п
__inference_loss_fn_0_73564П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_1_73580П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_2_73596П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_3_73612П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
9B7
#__inference_signature_wrapper_72767policy_input_0е
G__inference_activation_1_layer_call_and_return_conditional_losses_73255Z0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ ж
]__inference_tf_op_layer_Normal_1/sample/concat_layer_call_and_return_conditional_losses_73350E)в&
в
Ъ
К
inputs/0
к "в
К
0
Ъ ┘
d__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_and_return_conditional_losses_73382q?в<
5в2
0Ъ-
+К(
inputs/0                  
к ".в+
$К!
0                  
Ъ Е
/__inference_tf_op_layer_Exp_layer_call_fn_73517R6в3
,в)
'Ъ$
"К
inputs/0         
к "К         С
;__inference_tf_op_layer_clip_by_value_1_layer_call_fn_73495R6в3
,в)
'Ъ$
"К
inputs/0         
к "К         {
*__inference_activation_layer_call_fn_73172M0в-
&в#
!К
inputs         А
к "К         Ал
f__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_and_return_conditional_losses_73338A%в"
в
Ъ
К
inputs/0 
к "в
К
0
Ъ }
'__inference_dense_3_layer_call_fn_73436Rзи0в-
&в#
!К
inputs         А
к "К         Щ
C__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_fn_73471R6в3
,в)
'Ъ$
"К
inputs/0         
к "К         ▒
I__inference_tf_op_layer_Normal_1/sample/random_normal_layer_call_fn_73387d?в<
5в2
0Ъ-
+К(
inputs/0                  
к "!К                  ┤
o__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_and_return_conditional_losses_73442A%в"
в
Ъ
К
inputs/0 
к "в
К
0
Ъ ╣
C__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_fn_73507rMвJ
Cв@
>Ъ;
"К
inputs/0         
К
inputs/1
к "!К                  ╢
B__inference_model_1_layer_call_and_return_conditional_losses_72910p*,23>@FGRSзи7в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ ╜
Z__inference_tf_op_layer_Normal_1/sample/add_layer_call_and_return_conditional_losses_73404_6в3
,в)
'Ъ$
"К
inputs/0         
к "%в"
К
0         
Ъ ╪
y__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_and_return_conditional_losses_73360[)в&
в
Ъ
К
inputs/0
к ".в+
$К!
0                  
Ъ в
L__inference_tf_op_layer_Shape_layer_call_and_return_conditional_losses_73304R6в3
,в)
'Ъ$
"К
inputs/0         
к "в
К
0
Ъ ╞
Z__inference_tf_op_layer_Normal_1/sample/mul_layer_call_and_return_conditional_losses_73393h?в<
5в2
0Ъ-
+К(
inputs/0                  
к "%в"
К
0         
Ъ Щ
T__inference_tf_op_layer_strided_slice_layer_call_and_return_conditional_losses_73317A)в&
в
Ъ
К
inputs/0
к "в

К
0 
Ъ Ц
'__inference_model_1_layer_call_fn_72716k*,23>@FGRSзи?в<
5в2
(К%
policy_input_0         
p 

 
к "К         Ю
?__inference_tf_op_layer_Normal_1/sample/mul_layer_call_fn_73398[?в<
5в2
0Ъ-
+К(
inputs/0                  
к "К         Г
K__inference_tf_op_layer_Normal_1/sample/concat/values_0_layer_call_fn_733434%в"
в
Ъ
К
inputs/0 
к "К░
N__inference_layer_normalization_layer_call_and_return_conditional_losses_73155^230в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ ░
^__inference_tf_op_layer_Normal_1/sample/random_normal/RandomStandardNormal_layer_call_fn_73365N)в&
в
Ъ
К
inputs/0
к "!К                  г
B__inference_dense_2_layer_call_and_return_conditional_losses_73270]RS0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ ┐
\__inference_tf_op_layer_clip_by_value/Minimum_layer_call_and_return_conditional_losses_73283_6в3
,в)
'Ъ$
"К
inputs/0         
к "%в"
К
0         
Ъ Ц
'__inference_model_1_layer_call_fn_72616k*,23>@FGRSзи?в<
5в2
(К%
policy_input_0         
p

 
к "К         П
9__inference_tf_op_layer_clip_by_value_layer_call_fn_73299R6в3
,в)
'Ъ$
"К
inputs/0         
к "К         ▓
/__inference_tf_op_layer_mul_layer_call_fn_73529cв`
YвV
TЪQ
+К(
inputs/0                  
"К
inputs/1         
к "К         :
__inference_loss_fn_3_73612@в

в 
к "К ▓
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_73243^FG0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ Ч
A__inference_tf_op_layer_clip_by_value/Minimum_layer_call_fn_73288R6в3
,в)
'Ъ$
"К
inputs/0         
к "К         О
'__inference_model_1_layer_call_fn_73084c*,23>@FGRSзи7в4
-в*
 К
inputs         
p 

 
к "К         ╢
B__inference_model_1_layer_call_and_return_conditional_losses_73050p*,23>@FGRSзи7в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ ~
B__inference_tf_op_layer_Normal_1/sample/concat_layer_call_fn_733558)в&
в
Ъ
К
inputs/0
к "К|
'__inference_dense_1_layer_call_fn_73221Q>@0в-
&в#
!К
inputs         А
к "К         АЕ
I__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_fn_734608)в&
в
Ъ
К
inputs/0
к "К╚
#__inference_signature_wrapper_72767а*,23>@FGRSзиIвF
в 
?к<
:
policy_input_0(К%
policy_input_0         "Cк@
>
tf_op_layer_Tanh*К'
tf_op_layer_Tanh         М
T__inference_tf_op_layer_Normal_1/sample/expand_to_vector/Reshape_layer_call_fn_734474%в"
в
Ъ
К
inputs/0 
к "Кz
1__inference_tf_op_layer_Shape_layer_call_fn_73309E6в3
,в)
'Ъ$
"К
inputs/0         
к "К:
__inference_loss_fn_2_73596>в

в 
к "К :
__inference_loss_fn_0_73564*в

в 
к "К г
E__inference_activation_layer_call_and_return_conditional_losses_73167Z0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ К
5__inference_layer_normalization_1_layer_call_fn_73250QFG0в-
&в#
!К
inputs         А
к "К         А:
__inference_loss_fn_1_73580,в

в 
к "К д
B__inference_dense_1_layer_call_and_return_conditional_losses_73214^>@0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ {
'__inference_dense_2_layer_call_fn_73277PRS0в-
&в#
!К
inputs         А
к "К         ╥
J__inference_tf_op_layer_add_layer_call_and_return_conditional_losses_73535ГZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "%в"
К
0         
Ъ ▌
h__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_and_return_conditional_losses_73371q?в<
5в2
0Ъ-
+К(
inputs/0                  
к ".в+
$К!
0                  
Ъ ╛
B__inference_model_1_layer_call_and_return_conditional_losses_72436x*,23>@FGRSзи?в<
5в2
(К%
policy_input_0         
p

 
к "%в"
К
0         
Ъ │
 __inference__wrapped_model_71639О*,23>@FGRSзи7в4
-в*
(К%
policy_input_0         
к "Cк@
>
tf_op_layer_Tanh*К'
tf_op_layer_Tanh         б
@__inference_dense_layer_call_and_return_conditional_losses_73126]*,/в,
%в"
 К
inputs         
к "&в#
К
0         А
Ъ ╣
V__inference_tf_op_layer_clip_by_value_1_layer_call_and_return_conditional_losses_73490_6в3
,в)
'Ъ$
"К
inputs/0         
к "%в"
К
0         
Ъ █
J__inference_tf_op_layer_mul_layer_call_and_return_conditional_losses_73523Мcв`
YвV
TЪQ
+К(
inputs/0                  
"К
inputs/1         
к "%в"
К
0         
Ъ О
'__inference_model_1_layer_call_fn_73067c*,23>@FGRSзи7в4
-в*
 К
inputs         
p

 
к "К         Ь
[__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_and_return_conditional_losses_73328=%в"
в
Ъ
К
inputs/0 
к "в

К
0 
Ъ с
^__inference_tf_op_layer_Normal_1/sample/Reshape_layer_call_and_return_conditional_losses_73501MвJ
Cв@
>Ъ;
"К
inputs/0         
К
inputs/1
к ".в+
$К!
0                  
Ъ е
B__inference_dense_3_layer_call_and_return_conditional_losses_73429_зи0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ }
,__inference_activation_1_layer_call_fn_73260M0в-
&в#
!К
inputs         А
к "К         Ао
K__inference_tf_op_layer_Tanh_layer_call_and_return_conditional_losses_73546_6в3
,в)
'Ъ$
"К
inputs/0         
к "%в"
К
0         
Ъ ┴
^__inference_tf_op_layer_clip_by_value_1/Minimum_layer_call_and_return_conditional_losses_73466_6в3
,в)
'Ъ$
"К
inputs/0         
к "%в"
К
0         
Ъ ╡
M__inference_tf_op_layer_Normal_1/sample/random_normal/mul_layer_call_fn_73376d?в<
5в2
0Ъ-
+К(
inputs/0                  
к "!К                  Х
?__inference_tf_op_layer_Normal_1/sample/add_layer_call_fn_73409R6в3
,в)
'Ъ$
"К
inputs/0         
к "К         ┐
___inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_and_return_conditional_losses_73478\@в=
6в3
1Ъ.
К
inputs/0
К
inputs/1
к "в
К
0
Ъ М
C__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_fn_73419E6в3
,в)
'Ъ$
"К
inputs/0         
к "К╛
B__inference_model_1_layer_call_and_return_conditional_losses_72518x*,23>@FGRSзи?в<
5в2
(К%
policy_input_0         
p 

 
к "%в"
К
0         
Ъ q
9__inference_tf_op_layer_strided_slice_layer_call_fn_733224)в&
в
Ъ
К
inputs/0
к "К y
%__inference_dense_layer_call_fn_73133P*,/в,
%в"
 К
inputs         
к "К         АЧ
D__inference_tf_op_layer_Normal_1/sample/concat_1_layer_call_fn_73484O@в=
6в3
1Ъ.
К
inputs/0
К
inputs/1
к "Кн
J__inference_tf_op_layer_Exp_layer_call_and_return_conditional_losses_73512_6в3
,в)
'Ъ$
"К
inputs/0         
к "%в"
К
0         
Ъ t
@__inference_tf_op_layer_Normal_1/sample/Prod_layer_call_fn_733330%в"
в
Ъ
К
inputs/0 
к "К ╖
T__inference_tf_op_layer_clip_by_value_layer_call_and_return_conditional_losses_73294_6в3
,в)
'Ъ$
"К
inputs/0         
к "%в"
К
0         
Ъ й
/__inference_tf_op_layer_add_layer_call_fn_73541vZвW
PвM
KЪH
"К
inputs/0         
"К
inputs/1         
к "К         И
3__inference_layer_normalization_layer_call_fn_73162Q230в-
&в#
!К
inputs         А
к "К         Ан
d__inference_tf_op_layer_Normal_1/sample/strided_slice_layer_call_and_return_conditional_losses_73455E)в&
в
Ъ
К
inputs/0
к "в
К
0
Ъ ┤
^__inference_tf_op_layer_Normal_1/sample/Shape_2_layer_call_and_return_conditional_losses_73414R6в3
,в)
'Ъ$
"К
inputs/0         
к "в
К
0
Ъ Ж
0__inference_tf_op_layer_Tanh_layer_call_fn_73551R6в3
,в)
'Ъ$
"К
inputs/0         
к "К         
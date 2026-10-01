"""
Unversioned property data: how cooked Unreal 5 packages store an object's
properties.

Instead of naming each property, a cooked export starts with a small header:
runs of "skip so many, then so many present" over the class's property list,
and a mask saying which of the present ones are zero (and so carry no bytes).
The values follow in order, each in its type's own encoding. Reading it needs
the class's property list - its *schema* - in exactly the order the engine
builds it: the class's own properties first, then its parent's, each static
array one slot per element, editor-only properties left out.

The schemas below are that list for the few engine classes a course needs
(scene and static-mesh components, the virtual texture volume) and the structs
they hold. They are property names and types only, derived from the Unreal
Engine 5.6.1 headers (which Epic makes available to anyone with a linked
account; no engine code is included), and checked against the shipped data of
the one game they are used for: every static-mesh and scene component of its
sixteen levels decodes, 32,310 and 10,906 of them. That build carries two
property slots in `UPrimitiveComponent` that the engine's headers do not
(`CachedMaxDrawDistance` sits where the header puts it, `RelativeLocation`
two slots later), and one more in the volume, marked `(unknown)` here: a value
present in one of them is an error, not a guess. The instanced components'
schemas are the headers' too, but that build has rearranged those classes, so
the course export does not read their properties (see `acrally.py`).

A game patch can move slots. `read_properties` raises rather than mis-read,
and the course export counts how many components decoded and says so.
"""

from __future__ import annotations

import re
import struct


class PropertyError(Exception):
    """A value this cannot read: an unknown type, or a slot past the schema."""


SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    'UActorComponent': (
        ('PrimaryComponentTick', 'FActorComponentTickFunction'),
        ('ComponentTags', 'TArray<FName>'),
        ('AssetUserData', 'TArray<TObjectPtr<UAssetUserData>>'),
        ('UCSSerializationIndex', 'int32'),
        ('bNetAddressable', 'uint8'),
        ('bReplicateUsingRegisteredSubObjectList', 'uint8'),
        ('bReplicates', 'uint8'),
        ('bAutoActivate', 'uint8'),
        ('bIsActive', 'uint8'),
        ('bEditableWhenInherited', 'uint8'),
        ('bCanEverAffectNavigation', 'uint8'),
        ('bIsEditorOnly', 'uint8'),
        ('CreationMethod', 'EComponentCreationMethod'),
        ('OnComponentActivated', 'FActorComponentActivatedSignature'),
        ('OnComponentDeactivated', 'FActorComponentDeactivateSignature'),
    ),
    'USceneComponent': (
        ('PhysicsVolume', 'TWeakObjectPtr<APhysicsVolume>'),
        ('AttachParent', 'TObjectPtr<USceneComponent>'),
        ('AttachSocketName', 'FName'),
        ('AttachChildren', 'TArray<TObjectPtr<USceneComponent>>'),
        ('ClientAttachedChildren', 'TArray<TObjectPtr<USceneComponent>>'),
        ('RelativeLocation', 'FVector'),
        ('RelativeRotation', 'FRotator'),
        ('RelativeScale3D', 'FVector'),
        ('ComponentVelocity', 'FVector'),
        ('bComponentToWorldUpdated', 'uint8'),
        ('bAbsoluteLocation', 'uint8'),
        ('bAbsoluteRotation', 'uint8'),
        ('bAbsoluteScale', 'uint8'),
        ('bVisible', 'uint8'),
        ('bShouldBeAttached', 'uint8'),
        ('bShouldSnapLocationWhenAttached', 'uint8'),
        ('bShouldSnapRotationWhenAttached', 'uint8'),
        ('bShouldSnapScaleWhenAttached', 'uint8'),
        ('bShouldUpdatePhysicsVolume', 'uint8'),
        ('bHiddenInGame', 'uint8'),
        ('bBoundsChangeTriggersStreamingDataRebuild', 'uint8'),
        ('bUseAttachParentBound', 'uint8'),
        ('bComputeFastLocalBounds', 'uint8'),
        ('bComputeBoundsOnceForGame', 'uint8'),
        ('bComputedBoundsOnceForGame', 'uint8'),
        ('bIsNotRenderAttachmentRoot', 'uint8'),
        ('Mobility', 'TEnumAsByte<EComponentMobility::Type>'),
        ('DetailMode', 'TEnumAsByte<EDetailMode>'),
        ('PhysicsVolumeChangedDelegate', 'FPhysicsVolumeChanged'),
    ),
    'UPrimitiveComponent': (
        ('MinDrawDistance', 'float'),
        ('LDMaxDrawDistance', 'float'),
        ('CachedMaxDrawDistance', 'float'),
        ('DepthPriorityGroup', 'TEnumAsByte<ESceneDepthPriorityGroup>'),
        ('ViewOwnerDepthPriorityGroup', 'TEnumAsByte<ESceneDepthPriorityGroup>'),
        ('IndirectLightingCacheQuality', 'TEnumAsByte<EIndirectLightingCacheQuality>'),
        ('LightmapType', 'ELightmapType'),
        ('HLODBatchingPolicy', 'EHLODBatchingPolicy'),
        ('ShadowCacheInvalidationBehavior', 'EShadowCacheInvalidationBehavior'),
        ('bEnableAutoLODGeneration', 'uint8'),
        ('bIsActorTextureStreamingBuiltData', 'uint8'),
        ('bIsValidTextureStreamingBuiltData', 'uint8'),
        ('bNeverDistanceCull', 'uint8'),
        ('bAlwaysCreatePhysicsState', 'uint8'),
        ('bGenerateOverlapEvents', 'uint8'),
        ('(unknown)', 'unknown'),
        ('bMultiBodyOverlap', 'uint8'),
        ('bTraceComplexOnMove', 'uint8'),
        ('bReturnMaterialOnMove', 'uint8'),
        ('bUseViewOwnerDepthPriorityGroup', 'uint8'),
        ('bAllowCullDistanceVolume', 'uint8'),
        ('bVisibleInReflectionCaptures', 'uint8'),
        ('bVisibleInRealTimeSkyCaptures', 'uint8'),
        ('bVisibleInRayTracing', 'uint8'),
        ('bRenderInMainPass', 'uint8'),
        ('bRenderInDepthPass', 'uint8'),
        ('bReceivesDecals', 'uint8'),
        ('bHoldout', 'uint8'),
        ('bOwnerNoSee', 'uint8'),
        ('bOnlyOwnerSee', 'uint8'),
        ('bTreatAsBackgroundForOcclusion', 'uint8'),
        ('bUseAsOccluder', 'uint8'),
        ('bSelectable', 'uint8'),
        ('bWantsEditorEffects', 'uint8'),
        ('(unknown)', 'unknown'),
        ('bForceMipStreaming', 'uint8'),
        ('bHasPerInstanceHitProxies', 'uint8'),
        ('CastShadow', 'uint8'),
        ('bEmissiveLightSource', 'uint8'),
        ('bAffectDynamicIndirectLighting', 'uint8'),
        ('bAffectIndirectLightingWhileHidden', 'uint8'),
        ('bAffectDistanceFieldLighting', 'uint8'),
        ('bCastDynamicShadow', 'uint8'),
        ('bCastStaticShadow', 'uint8'),
        ('bCastVolumetricTranslucentShadow', 'uint8'),
        ('bCastContactShadow', 'uint8'),
        ('bSelfShadowOnly', 'uint8'),
        ('bCastFarShadow', 'uint8'),
        ('bCastInsetShadow', 'uint8'),
        ('bCastCinematicShadow', 'uint8'),
        ('bCastHiddenShadow', 'uint8'),
        ('bCastShadowAsTwoSided', 'uint8'),
        ('bLightAsIfStatic_DEPRECATED', 'uint8'),
        ('bLightAttachmentsAsGroup', 'uint8'),
        ('bExcludeFromLightAttachmentGroup', 'uint8'),
        ('bReceiveMobileCSMShadows', 'uint8'),
        ('bSingleSampleShadowFromStationaryLights', 'uint8'),
        ('bIgnoreRadialImpulse', 'uint8'),
        ('bIgnoreRadialForce', 'uint8'),
        ('bApplyImpulseOnDamage', 'uint8'),
        ('bReplicatePhysicsToAutonomousProxy', 'uint8'),
        ('bFillCollisionUnderneathForNavmesh', 'uint8'),
        ('bRasterizeAsFilledConvexVolume', 'uint8'),
        ('AlwaysLoadOnClient', 'uint8'),
        ('AlwaysLoadOnServer', 'uint8'),
        ('bUseEditorCompositing', 'uint8'),
        ('bIsBeingMovedByEditor', 'uint8'),
        ('bRenderCustomDepth', 'uint8'),
        ('bVisibleInSceneCaptureOnly', 'uint8'),
        ('bHiddenInSceneCapture', 'uint8'),
        ('bRayTracingFarField', 'uint8'),
        ('FirstPersonPrimitiveType', 'EFirstPersonPrimitiveType'),
        ('bHasNoStreamableTextures', 'uint8'),
        ('bStaticWhenNotMoveable', 'uint8'),
        ('bHasCustomNavigableGeometry', 'TEnumAsByte<EHasCustomNavigableGeometry::Type>'),
        ('CanCharacterStepUpOn', 'TEnumAsByte<ECanBeCharacterBase>'),
        ('LightingChannels', 'FLightingChannels'),
        ('RayTracingGroupCullingPriority', 'ERayTracingGroupCullingPriority'),
        ('CustomDepthStencilWriteMask', 'ERendererStencilMask'),
        ('ExcludeFromHLODLevels', 'uint8'),
        ('RayTracingGroupId', 'int32'),
        ('VisibilityId', 'int32'),
        ('CustomDepthStencilValue', 'int32'),
        ('CustomPrimitiveData', 'FCustomPrimitiveData'),
        ('CustomPrimitiveDataInternal', 'FCustomPrimitiveData'),
        ('TranslucencySortPriority', 'int32'),
        ('TranslucencySortDistanceOffset', 'float'),
        ('RuntimeVirtualTextures', 'TArray<TObjectPtr<URuntimeVirtualTexture>>'),
        ('VirtualTextureLodBias', 'int8'),
        ('VirtualTextureCullMips', 'int8'),
        ('VirtualTextureMinCoverage', 'int8'),
        ('VirtualTextureRenderPassType', 'ERuntimeVirtualTextureMainPassType'),
        ('BoundsScale', 'float'),
        ('MoveIgnoreActors', 'TArray<TObjectPtr<AActor>>'),
        ('MoveIgnoreComponents', 'TArray<TObjectPtr<UPrimitiveComponent>>'),
        ('BodyInstance', 'FBodyInstance'),
        ('OnComponentHit', 'FComponentHitSignature'),
        ('OnComponentBeginOverlap', 'FComponentBeginOverlapSignature'),
        ('OnComponentEndOverlap', 'FComponentEndOverlapSignature'),
        ('OnComponentWake', 'FComponentWakeSignature'),
        ('OnComponentSleep', 'FComponentSleepSignature'),
        ('OnComponentPhysicsStateChanged', 'FComponentPhysicsStateChanged'),
        ('OnBeginCursorOver', 'FComponentBeginCursorOverSignature'),
        ('OnEndCursorOver', 'FComponentEndCursorOverSignature'),
        ('OnClicked', 'FComponentOnClickedSignature'),
        ('OnReleased', 'FComponentOnReleasedSignature'),
        ('OnInputTouchBegin', 'FComponentOnInputTouchBeginSignature'),
        ('OnInputTouchEnd', 'FComponentOnInputTouchEndSignature'),
        ('OnInputTouchEnter', 'FComponentBeginTouchOverSignature'),
        ('OnInputTouchLeave', 'FComponentEndTouchOverSignature'),
        ('LODParentPrimitive', 'TObjectPtr<UPrimitiveComponent>'),
    ),
    'UMeshComponent': (
        ('OverrideMaterials', 'TArray<TObjectPtr<UMaterialInterface>>'),
        ('OverlayMaterial', 'TObjectPtr<UMaterialInterface>'),
        ('OverlayMaterialMaxDrawDistance', 'float'),
        ('MaterialSlotsOverlayMaterial', 'TArray<TObjectPtr<UMaterialInterface>>'),
        ('bEnableMaterialParameterCaching', 'uint8'),
    ),
    'UStaticMeshComponent': (
        ('ForcedLodModel', 'int32'),
        ('MinLOD', 'int32'),
        ('SubDivisionStepSize', 'int32'),
        ('WireframeColorOverride', 'FColor'),
        ('StaticMesh', 'TObjectPtr<UStaticMesh>'),
        ('WorldPositionOffsetDisableDistance', 'int32'),
        ('bForceNaniteForMasked', 'uint8'),
        ('bDisallowNanite', 'uint8'),
        ('bForceDisableNanite', 'uint8'),
        ('bEvaluateWorldPositionOffset', 'uint8'),
        ('bWorldPositionOffsetWritesVelocity', 'uint8'),
        ('bEvaluateWorldPositionOffsetInRayTracing', 'uint8'),
        ('bOverrideWireframeColor', 'uint8'),
        ('bOverrideMinLOD', 'uint8'),
        ('bOverrideNavigationExport', 'uint8'),
        ('bForceNavigationObstacle', 'uint8'),
        ('bDisallowMeshPaintPerInstance_DEPRECATED', 'uint8'),
        ('bIgnoreInstanceForTextureStreaming', 'uint8'),
        ('bOverrideLightMapRes', 'uint8'),
        ('bCastDistanceFieldIndirectShadow', 'uint8'),
        ('bOverrideDistanceFieldSelfShadowBias', 'uint8'),
        ('bUseSubDivisions', 'uint8'),
        ('bUseDefaultCollision', 'uint8'),
        ('bSortTriangles', 'uint8'),
        ('bReverseCulling', 'uint8'),
        ('bEnableVertexColorMeshPainting', 'uint8'),
        ('bEnableTextureColorMeshPainting', 'uint8'),
        ('bOverrideMeshPaintTextureCoordinateIndex', 'uint8'),
        ('bOverrideMeshPaintTextureResolution', 'uint8'),
        ('OverriddenMeshPaintTextureCoordinateIndex', 'int32'),
        ('OverriddenMeshPaintTextureResolution', 'int32'),
        ('OverriddenLightMapRes', 'int32'),
        ('MeshPaintTextureCooked', 'TObjectPtr<UTexture>'),
        ('MeshPaintTextureOverride', 'TObjectPtr<UTexture>'),
        ('MaterialCacheTexture', 'TObjectPtr<UTexture>'),
        ('DistanceFieldIndirectShadowMinVisibility', 'float'),
        ('DistanceFieldSelfShadowBias', 'float'),
        ('StreamingDistanceMultiplier', 'float'),
        ('NanitePixelProgrammableDistance', 'float'),
        ('LODData', 'TArray<FStaticMeshComponentLODInfo>'),
        ('StreamingTextureData', 'TArray<FStreamingTextureBuildInfo>'),
        ('LightmassSettings', 'FLightmassPrimitiveSettings'),
    ),
    'UInstancedStaticMeshComponent': (
        ('PerInstanceSMData', 'TArray<FInstancedStaticMeshInstanceData>'),
        ('PerInstancePrevTransform', 'TArray<FMatrix>'),
        ('PreviousComponentTransform', 'FTransform'),
        ('NumCustomDataFloats', 'int32'),
        ('InstancingRandomSeed', 'int32'),
        ('PerInstanceSMCustomData', 'TArray<float>'),
        ('AdditionalRandomSeeds', 'TArray<FInstancedStaticMeshRandomSeed>'),
        ('InstanceLODDistanceScale', 'float'),
        ('InstanceMinDrawDistance', 'int32'),
        ('InstanceStartCullDistance', 'int32'),
        ('InstanceEndCullDistance', 'int32'),
        ('bUseGpuLodSelection', 'uint8'),
        ('bInheritPerInstanceData', 'uint8'),
        ('bDisableCollision', 'uint8'),
        ('InstanceReorderTable', 'TArray<int32>'),
        ('NumPendingLightmaps', 'int32'),
        ('CachedMappings', 'TArray<FInstancedStaticMeshMappingInfo>'),
        ('CachedBounds[0]', 'FBoundsCacheElement'),
        ('CachedBounds[1]', 'FBoundsCacheElement'),
        ('CachedBounds[2]', 'FBoundsCacheElement'),
    ),
    'UHierarchicalInstancedStaticMeshComponent': (
        ('bUseTranslatedInstanceSpace', 'uint8'),
        ('TranslatedInstanceSpaceOrigin', 'FVector'),
        ('SortedInstances', 'TArray<int32>'),
        ('NumBuiltInstances', 'int32'),
        ('BuiltInstanceBounds', 'FBox'),
        ('UnbuiltInstanceBounds', 'FBox'),
        ('UnbuiltInstanceBoundsList', 'TArray<FBox>'),
        ('bEnableDensityScaling', 'uint8'),
        ('OcclusionLayerNumNodes', 'int32'),
        ('CacheMeshExtendedBounds', 'FBoxSphereBounds'),
        ('InstanceCountToRender', 'int32'),
    ),
    'UFoliageInstancedStaticMeshComponent': (
        ('OnInstanceTakePointDamage', 'FInstancePointDamageSignature'),
        ('OnInstanceTakeRadialDamage', 'FInstanceRadialDamageSignature'),
        ('bEnableDiscardOnLoad', 'bool'),
        ('GenerationGuid', 'FGuid'),
    ),
    'FBodyInstance': (
        ('PositionSolverIterationCount', 'uint8'),
        ('VelocitySolverIterationCount', 'uint8'),
        ('ProjectionSolverIterationCount', 'uint8'),
        ('ObjectType', 'TEnumAsByte<ECollisionChannel>'),
        ('CollisionEnabled', 'TEnumAsByte<ECollisionEnabled::Type>'),
        ('SleepFamily', 'ESleepFamily'),
        ('DOFMode', 'TEnumAsByte<EDOFMode::Type>'),
        ('bUseCCD', 'uint8'),
        ('bUseMACD', 'uint8'),
        ('bIgnoreAnalyticCollisions', 'uint8'),
        ('bNotifyRigidBodyCollision', 'uint8'),
        ('bSmoothEdgeCollisions', 'uint8'),
        ('bLockTranslation', 'uint8'),
        ('bLockRotation', 'uint8'),
        ('bLockXTranslation', 'uint8'),
        ('bLockYTranslation', 'uint8'),
        ('bLockZTranslation', 'uint8'),
        ('bLockXRotation', 'uint8'),
        ('bLockYRotation', 'uint8'),
        ('bLockZRotation', 'uint8'),
        ('bOverrideMaxAngularVelocity', 'uint8'),
        ('bOverrideMaxDepenetrationVelocity', 'uint8'),
        ('bOverrideWalkableSlopeOnInstance', 'uint8'),
        ('bInterpolateWhenSubStepping', 'uint8'),
        ('bInertiaConditioning', 'uint8'),
        ('bOneWayInteraction', 'uint8'),
        ('bOverrideSolverAsyncDeltaTime', 'uint8'),
        ('SolverAsyncDeltaTime', 'float'),
        ('CollisionProfileName', 'FName'),
        ('CollisionResponses', 'FCollisionResponse'),
        ('MaxDepenetrationVelocity', 'float'),
        ('MassInKgOverride', 'float'),
        ('LinearDamping', 'float'),
        ('AngularDamping', 'float'),
        ('CustomDOFPlaneNormal', 'FVector'),
        ('COMNudge', 'FVector'),
        ('MassScale', 'float'),
        ('GravityGroupIndex', 'uint8'),
        ('InertiaTensorScale', 'FVector'),
        ('WalkableSlopeOverride', 'FWalkableSlopeOverride'),
        ('PhysMaterialOverride', 'TObjectPtr<UPhysicalMaterial>'),
        ('MaxAngularVelocity', 'float'),
        ('CustomSleepThresholdMultiplier', 'float'),
        ('StabilizationThresholdMultiplier', 'float'),
        ('PhysicsBlendWeight', 'float'),
    ),
    'FBodyInstanceCore': (
        ('bSimulatePhysics', 'uint8'),
        ('bOverrideMass', 'uint8'),
        ('bEnableGravity', 'uint8'),
        ('bUpdateKinematicFromSimulation', 'uint8'),
        ('bGyroscopicTorqueEnabled', 'uint8'),
        ('bAutoWeld', 'uint8'),
        ('bStartAwake', 'uint8'),
        ('bGenerateWakeEvents', 'uint8'),
        ('bUpdateMassWhenScaleChanges', 'uint8'),
    ),
    'FCollisionResponse': (
        ('ResponseToChannels', 'FCollisionResponseContainer'),
        ('ResponseArray', 'TArray<FResponseChannel>'),
    ),
    'FCollisionResponseContainer': (
        ('WorldStatic', 'TEnumAsByte<ECollisionResponse>'),
        ('WorldDynamic', 'TEnumAsByte<ECollisionResponse>'),
        ('Pawn', 'TEnumAsByte<ECollisionResponse>'),
        ('Visibility', 'TEnumAsByte<ECollisionResponse>'),
        ('Camera', 'TEnumAsByte<ECollisionResponse>'),
        ('PhysicsBody', 'TEnumAsByte<ECollisionResponse>'),
        ('Vehicle', 'TEnumAsByte<ECollisionResponse>'),
        ('Destructible', 'TEnumAsByte<ECollisionResponse>'),
        ('EngineTraceChannel1', 'TEnumAsByte<ECollisionResponse>'),
        ('EngineTraceChannel2', 'TEnumAsByte<ECollisionResponse>'),
        ('EngineTraceChannel3', 'TEnumAsByte<ECollisionResponse>'),
        ('EngineTraceChannel4', 'TEnumAsByte<ECollisionResponse>'),
        ('EngineTraceChannel5', 'TEnumAsByte<ECollisionResponse>'),
        ('EngineTraceChannel6', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel1', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel2', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel3', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel4', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel5', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel6', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel7', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel8', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel9', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel10', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel11', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel12', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel13', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel14', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel15', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel16', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel17', 'TEnumAsByte<ECollisionResponse>'),
        ('GameTraceChannel18', 'TEnumAsByte<ECollisionResponse>'),
    ),
    'FResponseChannel': (
        ('Channel', 'FName'),
        ('Response', 'TEnumAsByte<ECollisionResponse>'),
    ),
    'FStreamingTextureBuildInfo': (
        ('PackedRelativeBox', 'uint32'),
        ('TextureLevelIndex', 'int32'),
        ('TexelFactor', 'float'),
    ),
    'FBoundsCacheElement': (
        ('bIsValid', 'bool'),
        ('Hash', 'uint32'),
        ('Value', 'FBoxSphereBounds'),
    ),
    'FInstancedStaticMeshRandomSeed': (
        ('StartInstanceIndex', 'int32'),
        ('RandomSeed', 'int32'),
    ),
    'FCustomPrimitiveData': (
        ('Data', 'TArray<float>'),
    ),
}

#: The volume a baked virtual texture covers. Twenty properties in the public
#: header and one more in this build, before its scene component's: every
#: stage's volume reads with it and leaves exactly the scene component's
#: eight-byte native tail.
SCHEMAS["URuntimeVirtualTextureComponent"] = (
    ("BoundsAlignActor", "TSoftObjectPtr<AActor>"),
    ("bSetBoundsButton", "bool"),
    ("bSnapBoundsToLandscape", "bool"),
    ("ExpandBounds", "float"),
    ("VirtualTexture", "TObjectPtr<URuntimeVirtualTexture>"),
    ("EnableInGamePerPlatform", "FPerPlatformBool"),
    ("bEnableForNaniteOnly", "bool"),
    ("bUseMinMaterialQuality", "bool"),
    ("MinInGameMaterialQuality", "ERuntimeVirtualTextureMaterialQuality"),
    ("bEnableScalability", "bool"),
    ("ScalabilityGroup", "uint32"),
    ("bHidePrimitives", "bool"),
    ("StreamingTexture", "TObjectPtr<UVirtualTextureBuilder>"),
    ("StreamLowMips", "int32"),
    ("bBuildStreamingMipsButton", "bool"),
    ("LossyCompressionAmount", "TEnumAsByte<ETextureLossyCompressionAmount>"),
    ("bUseStreamingMipsFixedColor", "bool"),
    ("StreamingMipsFixedColor", "FLinearColor"),
    ("bUseStreamingMipsOnly", "bool"),
    ("UseStreamingMipsInEditorMode", "ERuntimeVirtualTextureUseStreamingMipsInEditorMode"),
    ("(unknown)", "unknown"),
)

_SMC = ("UStaticMeshComponent", "UMeshComponent", "UPrimitiveComponent", "USceneComponent", "UActorComponent")
_ISM = ("UInstancedStaticMeshComponent", *_SMC)
_HISM = ("UHierarchicalInstancedStaticMeshComponent", *_ISM)

#: Class paths to the schemas their properties are read with, own class first.
CHAINS: dict[str, tuple[str, ...]] = {
    "/Script/Engine.SceneComponent": ("USceneComponent", "UActorComponent"),
    "/Script/Engine.StaticMeshComponent": _SMC,
    "/Script/Engine.InstancedStaticMeshComponent": _ISM,
    "/Script/Engine.HierarchicalInstancedStaticMeshComponent": _HISM,
    "/Script/Foliage.FoliageInstancedStaticMeshComponent": ("UFoliageInstancedStaticMeshComponent", *_HISM),
    "/Script/Engine.RuntimeVirtualTextureComponent": ("URuntimeVirtualTextureComponent", "USceneComponent", "UActorComponent"),
}

#: Structs that extend another, own properties first as for classes.
STRUCT_CHAINS: dict[str, tuple[str, ...]] = {
    "FBodyInstance": ("FBodyInstance", "FBodyInstanceCore"),
}

#: Types with a fixed size, and how to read them.
_FIXED = {
    "bool": ("<?", 1), "int8": ("<b", 1), "uint8": ("<B", 1), "int16": ("<h", 2), "uint16": ("<H", 2),
    "int32": ("<i", 4), "uint32": ("<I", 4), "float": ("<f", 4), "int64": ("<q", 8), "uint64": ("<Q", 8),
    "double": ("<d", 8), "FColor": ("<4B", 4), "FGuid": ("<4I", 16), "FVector": ("<3d", 24),
    "FRotator": ("<3d", 24), "FVector3f": ("<3f", 12), "FVector2D": ("<2d", 16), "FLinearColor": ("<4f", 16),
    "FQuat": ("<4d", 32), "FBox": ("<6dB", 49), "FBoxSphereBounds": ("<7d", 56), "FMatrix": ("<16d", 128),
    "FTransform": ("<10d", 80), "FIntPoint": ("<2i", 8), "FIntVector": ("<3i", 12),
}
_OBJECT = re.compile(r"^(TObjectPtr|TWeakObjectPtr|TSubclassOf|TLazyObjectPtr)<.*>$|^\w+\s*\*$")
_ARRAY = re.compile(r"^TArray<(.*)>$")
_ENUM = re.compile(r"^(TEnumAsByte<.*>|E[A-Z]\w*(::Type)?)$")
#: Multicast delegates: an array of (object, function name).
_DELEGATE = re.compile(r"^F\w*(Signature|Changed|Delegate)$")


def schema(chain: tuple[str, ...]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name in chain:
        out.extend(SCHEMAS[name])
    return out


def read_header(data: bytes, at: int) -> tuple[list[tuple[int, bool]], int]:
    """The present properties, as (schema slot, is zero), and where the values start."""
    fragments = []
    while True:
        (word,) = struct.unpack_from("<H", data, at)
        at += 2
        fragments.append((word & 0x7F, bool(word & 0x80), word >> 9))
        if word & 0x100:
            break
    zero_count = sum(count for _skip, has_zero, count in fragments if has_zero)
    zeros: list[bool] = []
    if zero_count:
        size = 1 if zero_count <= 8 else 2 if zero_count <= 16 else 4 * ((zero_count + 31) // 32)
        raw = data[at : at + size]
        at += size
        zeros = [bool(raw[i // 8] >> (i % 8) & 1) for i in range(zero_count)]
    present = []
    slot = 0
    zero_at = 0
    for skip, has_zero, count in fragments:
        slot += skip
        for _ in range(count):
            is_zero = False
            if has_zero:
                is_zero = zeros[zero_at]
                zero_at += 1
            present.append((slot, is_zero))
            slot += 1
    return present, at


def _value(data: bytes, at: int, kind: str, names: list[str]):
    kind = kind.strip()
    fixed = _FIXED.get(kind)
    if fixed is not None:
        fmt, size = fixed
        values = struct.unpack_from(fmt, data, at)
        return (values[0] if len(values) == 1 else values), at + size
    if kind == "FName":
        index, number = struct.unpack_from("<II", data, at)
        name = names[index] if index < len(names) else f"#{index}"
        return (name if number == 0 else f"{name}_{number - 1}"), at + 8
    if _ENUM.match(kind):
        return data[at], at + 1
    if _OBJECT.match(kind):
        return struct.unpack_from("<i", data, at)[0], at + 4
    if kind.startswith(("TSoftObjectPtr<", "TSoftClassPtr<")) or kind in ("FSoftObjectPath", "FSoftClassPath"):
        # The asset's package and name, then a sub-path string.
        package, asset = _value(data, at, "FName", names)[0], _value(data, at + 8, "FName", names)[0]
        (length,) = struct.unpack_from("<i", data, at + 16)
        size = length if length >= 0 else -2 * length
        return f"{package}.{asset}", at + 20 + size
    array = _ARRAY.match(kind)
    if array:
        (count,) = struct.unpack_from("<i", data, at)
        at += 4
        if count < 0 or count > 50_000_000:
            raise PropertyError(f"implausible array length {count}")
        items = []
        for _ in range(count):
            item, at = _value(data, at, array.group(1), names)
            items.append(item)
        return items, at
    if kind in SCHEMAS:
        return read_struct(data, at, schema(STRUCT_CHAINS.get(kind, (kind,))), names)
    if _DELEGATE.match(kind):
        (count,) = struct.unpack_from("<i", data, at)
        return None, at + 4 + count * 12
    raise PropertyError(f"no reader for type {kind}")


def read_struct(data: bytes, at: int, slots, names: list[str]) -> tuple[dict, int]:
    """One property block - an object's, or a struct value's - by its schema."""
    present, at = read_header(data, at)
    out: dict = {}
    for slot, is_zero in present:
        if slot >= len(slots):
            raise PropertyError(f"slot {slot} is past the schema's {len(slots)}")
        name, kind = slots[slot]
        if is_zero:
            out[name] = 0
            continue
        if name == "(unknown)":
            raise PropertyError(f"a value in uncalibrated slot {slot}")
        out[name], at = _value(data, at, kind, names)
    return out, at


def read_properties(data: bytes, class_path: str, names: list[str]) -> tuple[dict, int]:
    """An export's properties by its class, and where its native data starts."""
    chain = CHAINS.get(class_path)
    if chain is None:
        raise PropertyError(f"no schema for {class_path}")
    return read_struct(data, 0, schema(chain), names)

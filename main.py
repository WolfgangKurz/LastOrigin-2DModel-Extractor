import sys
import os
import re
from struct import unpack

from PIL import Image, ImageDraw

import UnityPy
import numpy as np
import quaternion

from photoshop import Session
from photoshop.api._document import Document, LayerSet, ArtLayer

from matrix import Matrix3D
from bcolors import bcolors


def main(input: str) -> None:
    __dirname__ = os.path.dirname(__file__)
    outDir = os.path.dirname(input)

    inpName = os.path.basename(input)
    inpCharName = re.search(r"2dmodel_(.+)", inpName).group(1)

    env = UnityPy.load(input)

    # load dependencies
    for obj in env.objects:
        if obj.type.name == "AssetBundle":
            data = obj.read()

            for dep in data.m_Dependencies:
                if dep == "tbaricon":  # pass, it is too heavy
                    continue

                dep_path = os.path.join(os.path.dirname(input), dep)
                if os.path.exists(dep_path):  # try to load in same directory
                    env.load_file()

    env.load_file(os.path.join(__dirname__, "unity_builtin_extra"))

    SkipShaders = ["Sprites/Default"]
    RenameShaders = {
        "LastOne/LO_Sprite_loby_cha_full3Dmeshbase_Additional_Alpha": "additional-alpha",
        "Legacy Shaders/Particles/Additive": "additive",
        "Legacy Shaders/Particles/Additive (Soft)": "additive-soft",
        "Custom/Additive (Soft)": "additive-soft",
        "Legacy Shaders/Particles/Multiply": "multiply",
    }

    def unwrap(data):
        if str(type(data)) == "<class 'UnityPy.classes.PPtr.PPtr'>":
            return data.get_obj()
        return data

    def euler_from_quaternion(x, y, z, w):
        # https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
    
        return roll_x, pitch_y, yaw_z # in radians

    table = {}
    hasSprite = False
    # pre-init check
    for obj in env.objects:
        if obj.type.name == "Animator" or obj.type.name == "SkinnedMeshRenderer" or obj.type.name == "ParticleSystem" or obj.type.name == "MeshFilter" or obj.type.name == "MeshRenderer":
            # Animated 2DModel not available yet
            # If model is Spine-animated, try extract TextAssets and Textures, and load with Spine players(renderers)
            return
        elif obj.type.name == "Sprite":
            hasSprite = True

        table[obj.path_id] = obj

    if not hasSprite:
        return  # If bundle not contains any Sprite, it is invalid target

    tree = {}
    rootNode = {
        "text": "root",
        "childs": [],
    }

    obj = None  # make object tree
    for obj in env.objects:
        if obj.type.name == "GameObject":
            data = obj.read()

            if obj not in tree:
                currentNode = {
                    "gameObject": obj,
                    "text": data.m_Name,
                    "childs": [],
                }
                tree[obj] = currentNode
            else:
                currentNode = tree[obj]

            parentNode = rootNode

            if data.m_Transform is not None:
                m_Father = data.m_Transform.read().m_Father
                if m_Father.path_id != 0:
                    parentGameObject = unwrap(table[m_Father.path_id].read().m_GameObject)

                    if parentGameObject not in tree:
                        parentGameObjectData = parentGameObject.read()
                        parentGameObjectNode = {
                            "gameObject": parentGameObject,
                            "text": parentGameObjectData.m_Name,
                            "childs": [],
                        }
                        tree[parentGameObject] = parentGameObjectNode
                        del parentGameObjectData
                    else:
                        parentGameObjectNode = tree[parentGameObject]

                    parentNode = parentGameObjectNode
                    del parentGameObjectNode
                del m_Father
            parentNode["childs"].append(currentNode)

            del parentNode, currentNode, data

    if obj is None:
        return  # no objects

    del obj

    root = rootNode["childs"][0]
    del rootNode, tree

    listParts = []
    listBg = []
    listSwapActive = []
    listSwapInactive = []
    hasSwap = len(listSwapActive) > 0 or len(listSwapInactive) > 0
    orderTable: dict[int, list[LayerSet]] = {}

    # build list data
    for obj in env.objects:
        if obj.type.name == "MonoBehaviour":
            mono = obj.read()
            script = mono.m_Script.read()

            if script.name == "ActorPartsView":
                listParts = [p.read().m_Name for p in mono._listParts if p.path_id != 0]
                listBg = [p.read().m_Name for p in mono._listBg if p.path_id != 0]

                hasSwap = mono._isSwapParts != 0
                listSwapActive = [p.read().m_Name for p in mono._listSwapActiveObject if p.path_id != 0]
                listSwapInactive = [p.read().m_Name for p in mono._listSwapDeactiveObject if p.path_id != 0]

    passing = True
    temp_id = 0

    def InList(name: str) -> bool:
        return name in listParts or \
            name in listBg or \
            name in listSwapActive or \
            name in listSwapInactive

    def Process(ps: Session, document: Document, documentFillLayer: ArtLayer, target, matrices: list[np.ndarray] = [], _list = [False, False, False, False]) -> None:
        nonlocal listParts, listBg
        nonlocal orderTable, passing, temp_id

        desc = ps.ActionDescriptor

        # "tex balloon pos" = for Unit quote box base position
        if target["text"] == "tex balloon pos" or target["text"] == "Camera_Boundary":
            return None

        obj = target["gameObject"]
        data = obj.read()

        if not data.type_tree.m_IsActive and not InList(target["text"]):  # disabled GameObject
            return None

        _list[0] = _list[0] or (data.m_Name in listParts)
        _list[1] = _list[1] or (data.m_Name in listBg)
        _list[2] = _list[2] or (data.m_Name in listSwapActive)
        _list[3] = _list[3] or (data.m_Name in listSwapInactive)

        if _list[0] and (data.m_Name not in listParts): listParts.append(data.m_Name)
        if _list[1] and (data.m_Name not in listBg): listBg.append(data.m_Name)
        if _list[2] and (data.m_Name not in listSwapActive): listSwapActive.append(data.m_Name)
        if _list[3] and (data.m_Name not in listSwapInactive): listSwapInactive.append(data.m_Name)

        _matrices = matrices.copy()

        def getComponent(type: str, table, components):
            for component in components:
                if component.type.name == type:
                    return table[component.path_id].read()
            return None

        if not passing:
            # Transforms
            cd = getComponent("Transform", table, data.m_Components)
            if cd is not None:
                rot = quaternion.from_float_array([
                    cd.m_LocalRotation.W,
                    cd.m_LocalRotation.X,
                    cd.m_LocalRotation.Y,
                    cd.m_LocalRotation.Z,
                ])
                rot = quaternion.as_rotation_matrix(rot)

                _rot = np.identity(4)
                for i in range(3):
                    for j in range(3):
                        _rot[i][j] = rot[i][j]

                composed = Matrix3D.compose([
                    Matrix3D.translate(
                        cd.m_LocalPosition.X,
                        -cd.m_LocalPosition.Y,
                        cd.m_LocalPosition.Z
                    ),
                    Matrix3D.scale(
                        cd.m_LocalScale.X,
                        cd.m_LocalScale.Y,
                        cd.m_LocalScale.Z
                    ),
                    _rot
                    # Matrix3D.rotate(rot[0], rot[1], rot[2])
                ])

                _matrices.append(composed)

            # SpriteRenderer
            cd = getComponent("SpriteRenderer", table, data.m_Components)
            if cd is not None:
                orderId = cd.m_SortingOrder
                if orderId not in orderTable:
                    orderTable[orderId] = []

                if cd.m_Sprite.path_id != 0:
                    materials = [x.read() for x in cd.m_Materials]
                    shaders = [x.m_Shader.read().m_ParsedForm for x in materials]
                    shader = [
                        (RenameShaders[x.m_Name] if x.m_Name in RenameShaders else x.m_Name)
                        for x in shaders
                        if len(x.m_Name) > 0 and x.m_Name not in SkipShaders
                    ]

                    sprite = cd.m_Sprite.read()

                    atlas = sprite.m_RD
                    texture = atlas.texture.read()
                    _texture_image: Image = texture.image

                    #region make sprite image

                    # crop from texture
                    _sp: Image = _texture_image.crop((
                        sprite.m_Rect.x,
                        texture.image.height - sprite.m_Rect.y - sprite.m_Rect.height,
                        sprite.m_Rect.x + sprite.m_Rect.width,
                        texture.image.height - sprite.m_Rect.y,
                    ))

                    # rotate based settings
                    if atlas.settingsRaw.packed == 1:
                        _rotation = atlas.settingsRaw.packingRotation
                        if _rotation == 1: # flip horizontal
                            _sp = _sp.transpose(Image.FLIP_LEFT_RIGHT)
                        elif _rotation == 2: # flip vertical
                            _sp = _sp.transpose(Image.FLIP_TOP_BOTTOM)
                        elif _rotation == 3: # rotate 180deg
                            _sp = _sp.transpose(Image.ROTATE_180)
                        elif _rotation == 4: # rotate 90deg (in unity angle)
                            _sp = _sp.Transpose(Image.ROTATE_270) # so 270
                    
                    if atlas.settingsRaw.packingMode == 0: # tight
                        _mask = Image.new("1", _sp.size, color=0)
                        _draw = ImageDraw.ImageDraw(_mask)

                        def _get_triangles(_sprite):
                            m_RD = sprite.m_RD

                            # read the raw points
                            points = []
                            if hasattr(m_RD, "vertices"):  # 5.6 down
                                vertices = [v.pos for v in m_RD.vertices]
                                points = [vertices[index] for index in m_RD.indices]
                            else:  # 5.6 and up
                                m_Channel = m_RD.m_VertexData.m_Channels[0]  # kShaderChannelVertex
                                m_Stream = m_RD.m_VertexData.m_Streams[m_Channel.stream]

                                vertexData = m_RD.m_VertexData.m_DataSize
                                vertexDataPos = 0

                                indexData = m_RD.m_IndexBuffer
                                indexDataPos = 0

                                for subMesh in m_RD.m_SubMeshes:
                                    vertexDataPos = (
                                        m_Stream.offset
                                        + subMesh.firstVertex * m_Stream.stride
                                        + m_Channel.offset
                                    )

                                    vertices = []
                                    for _ in range(subMesh.vertexCount):
                                        vertices.append(
                                            unpack("<fff", vertexData[vertexDataPos:vertexDataPos+12])
                                        )
                                        vertexDataPos += m_Stream.stride

                                    indexDataPos = subMesh.firstByte
                                    for _ in range(subMesh.indexCount):
                                        points.append(
                                            vertices[
                                                unpack("<H", indexData[indexDataPos:indexDataPos+2])[0]
                                                - subMesh.firstVertex
                                            ]
                                        )
                                        indexDataPos += 2

                            # normalize the points
                            #  shift the whole point matrix into the positive space
                            #  multiply them with a factor to scale them to the image
                            min_x = min(p[0] for p in points)
                            min_y = min(p[1] for p in points)
                            factor = _sprite.m_PixelsToUnits
                            points = [
                                (
                                    (p[0] - min_x) * factor,
                                    (p[1] - min_y) * factor
                                )
                                for p in points
                            ]
                            return [points[i : i + 3] for i in range(0, len(points), 3)]

                        for _tri in _get_triangles(sprite):
                            _draw.polygon(_tri, fill=1)

                        _mask = _mask.transpose(Image.FLIP_TOP_BOTTOM) # flip vertically

                        # adjust texture rect offset
                        _ox = atlas.textureRectOffset.X
                        _oy = atlas.textureRectOffset.Y

                        __mask = Image.new(_mask.mode, _mask.size, color=0)
                        _draw = ImageDraw.ImageDraw(__mask)
                        _draw.bitmap((_ox, -_oy), _mask, fill=1)
                        _mask = __mask

                        if _sp.mode == "RGBA":
                            _empty = Image.new(_sp.mode, _sp.size, color=0)
                            _sp = Image.composite(_sp, _empty, _mask)
                        else:
                            _sp.putalpha(_mask)

                        if sprite.m_PixelsToUnits != 100:
                            _sp = _sp.resize(
                                (int(_sp.width * 100 / sprite.m_PixelsToUnits), int(_sp.height * 100 / sprite.m_PixelsToUnits)),
                                Image.LANCZOS
                            )
                    #endregion

                    # save into file temporary to load from photoshop
                    temp_id += 1
                    temp_path = os.path.join(outDir, f"_temp_{temp_id}.png")
                    _sp.save(temp_path)

                    # size of sprite
                    _w = sprite.m_Rect.width * 100 / sprite.m_PixelsToUnits
                    _h = sprite.m_Rect.height * 100 / sprite.m_PixelsToUnits

                    # expand canvas size at least sprite image size
                    if document.width < int(_w) or document.height < int(_h):
                        __w = max(int(_w), document.width)
                        __h = max(int(_h), document.height)
                        #  NOTE: Cannot call resizeCanvas with PS2025 (maybe, at least mine)
                        #  NOTE: So call JS, could be crash when working document is not active document
                        ps.app.doJavaScript(f"app.activeDocument.resizeCanvas({__w}, {__h}, AnchorPosition.MIDDLECENTER)")

                        # fill document-size layer to adjusting camera
                        document.activeLayer = documentFillLayer
                        document.selection.select(
                            [
                                [0, 0],
                                [__w, 0],
                                [__w, __h],
                                [0, __h],
                            ],
                            ps.SelectionType.ReplaceSelection,
                            0,
                            False
                        )
                        _color = ps.SolidColor()
                        _color.rgb.red = 0
                        _color.rgb.green = 0
                        _color.rgb.blue = 0
                        document.selection.fill(_color)
                        document.selection.deselect()

                        #  NOTE: fitLayersOnScreen is not for document, for active layer
                        ps.app.doJavaScript("app.runMenuItem(stringIDToTypeID('fitLayersOnScreen'));") # make zoom to fit document

                    # load sprite image into photoshop
                    desc.putPath(ps.app.charIDToTypeID("null"), temp_path)
                    ps.app.executeAction(ps.app.charIDToTypeID("Plc "), desc)
                    os.unlink(temp_path)

                    # size of document
                    _cx = document.width / 2
                    _cy = document.height / 2

                    # rename layer
                    layer: LayerSet = document.artLayers.getByIndex(0)
                    layer.name = data.m_Name

                    # opacity, tint
                    r, g, b, a = cd.m_Color.r, cd.m_Color.g, cd.m_Color.b, cd.m_Color.a
                    if a != 1:
                        layer.opacity = a * 100
                    if r + g + b != 3:
                        raise Exception("Tint not white")

                    # placeholder
                    _placeholder = document.artLayers.add()
                    _placeholder.name = "placeholder"

                    document.selection.select(
                        [
                            [
                                p[0] + _cx - _w / 2,
                                p[1] + _cy - _h / 2
                            ]
                            for p in [
                                [0, 0],
                                [_w, 0],
                                [_w, _h],
                                [0, _h],
                            ]
                        ],
                        ps.SelectionType.ReplaceSelection,
                        0,
                        False
                    )

                    _color = ps.SolidColor()
                    _color.rgb.red = 0
                    _color.rgb.green = 0
                    _color.rgb.blue = 0
                    document.selection.fill(_color)
                    document.selection.deselect()

                    # group
                    _group = document.layerSets.add()
                    _group.name = layer.name

                    _placeholder.move(_group, ps.ElementPlacement.PlaceInside)
                    layer.move(_group, ps.ElementPlacement.PlaceInside)
                    document.activeLayer = _group

                    layer: LayerSet = _group

                    # apply mix-blend from shader
                    for s in shader:
                        if isinstance(s, list):
                            layer.opacity = s[1] * 100
                            blend = s[0]
                        else:
                            blend = s

                        match blend:
                            case "linear-dodge":
                                layer.blendMode = ps.BlendMode.LinearDodge
                            case "multiply":
                                layer.blendMode = ps.BlendMode.Multiply
                            case _:
                                raise Exception("Unknown blendMode " + blend)

                    mat = Matrix3D.compose(_matrices) # 4x4 matrix

                    # resize and rotate
                    _s_x = np.linalg.norm(mat[0])
                    _s_y = np.linalg.norm(mat[1])
                    _p_x = mat[3,0]
                    _p_y = mat[3,1]

                    q = quaternion.from_rotation_matrix(mat)
                    r = euler_from_quaternion(q.x, q.y, q.z, q.w)
                    r = (np.rad2deg(r) + 360) % 360

                    # resize
                    if _s_x != 1 or _s_y != 1:
                        layer.resize(
                            _s_x * 100,
                            _s_y * 100,
                            anchor=ps.AnchorPosition.MiddleCenter
                        )

                    # flip
                    _flip_x = True if cd.m_FlipX else False
                    _flip_y = True if cd.m_FlipY else False
                    if _flip_x or _flip_y:
                        layer.resize(
                            100 * (-1 if _flip_x else 1),
                            100 * (-1 if _flip_y else 1),
                        )

                    # rotate
                    if r[2] != 0:
                        layer.rotate(-r[2], ps.AnchorPosition.MiddleCenter)

                    # translate
                    layer.translate(_p_x * 100, _p_y * 100)

                    # make transparent placeholder
                    _placeholder.fillOpacity = 0
                    _placeholder.opacity = 0

                    orderTable[orderId].append(layer)

        if passing:
            if data.m_Name.endswith("_root"):
                passing = False

        for c in target["childs"]:
            __list = [_list[0], _list[1], _list[2], _list[3]]
            Process(ps, document, documentFillLayer, c, _matrices, __list)

    rootName: str = root["text"]
    rootName = rootName.replace("_dam", "_Dam") # patch

    with Session() as ps:
        ps.app.preferences.rulerUnits = ps.Units.Pixels

        if True:  # Change this to False if want to use currently opening document
            document = ps.app.documents.add(
                256, 256,  # small size at first
                name=inpCharName,
                initialFill=ps.DocumentFill.Transparent
            )
        else:
            document = ps.active_document

        try:
            documentFillLayer = document.artLayers.add()
            documentFillLayer.name = "#DocumentFillLayer"

            Process(ps, document, documentFillLayer, root)
            
            documentFillLayer.remove()
            flatten_layers = sum(list(orderTable.values()), [])

            def _find_layer(name: str) -> LayerSet | None:
                nonlocal flatten_layers
                for layer in flatten_layers:
                    if layer.name == name:
                        return layer
                return None
            def _active_layer(name: str, active: bool) -> None:
                try:
                    _layer = _find_layer(name)
                    if _layer is not None:
                        _layer.visible = active
                except:
                    pass
            def active_Part(active: bool) -> None:
                if hasSwap:
                    for p in listSwapInactive: _active_layer(p, not active)
                    for p in listSwapActive: _active_layer(p, active)
                else:
                    for p in listParts:
                        _active_layer(p, active)
            def active_BG(active: bool) -> None:
                for p in listBg:
                    _active_layer(p, active)

            orders = list(orderTable.keys())
            orders.sort()
            orders = [x for x in orders if len(orderTable[x]) > 0]
            for order in orders:
                _layers = orderTable[order]
                _grp = document.layerSets.add()
                _grp.name = str(order)

                for layer in _layers:
                    _temp_layer = document.artLayers.add()
                    _temp_layer.move(_grp, ps.ElementPlacement.PlaceInside)
                    layer.move(_temp_layer, ps.ElementPlacement.PlaceBefore)
                    _temp_layer.remove()
                    # layer.move(_grp, ps.ElementPlacement.PlaceInside)

            document.reveal_all()  # expand to all layers' bound
            document.trim(ps.TrimType.TransparentPixels)

            _basename = rootName
            _postfix_base = "_"
            if input.endswith("_dam"):
                _postfix_base += "Dam"

            active_Part(True)
            active_BG(True)

            def _save(postfix: str) -> None:
                _postfix = _postfix_base + postfix
                filename = _basename + (_postfix if len(_postfix) > 1 else "") + ".psd"

                options = ps.PhotoshopSaveOptions()
                document.saveAs(
                    os.path.join(outDir, filename),
                    options,
                    True
                )
                print(f"File saved to {bcolors.cyan(filename)}")

            _save("")
            _history = document.activeHistoryState

            if len(listBg) > 0:
                document.activeHistoryState = _history
                active_BG(False)
                active_Part(True)

                document.trim(ps.TrimType.TransparentPixels)
                _save("B")

            if hasSwap or len(listParts) > 0:
                document.activeHistoryState = _history
                active_BG(True)
                active_Part(False)

                document.trim(ps.TrimType.TransparentPixels)
                _save("S")

            if (hasSwap or len(listParts) > 0) and (len(listBg) > 0):
                document.activeHistoryState = _history
                active_BG(False)
                active_Part(False)

                document.trim(ps.TrimType.TransparentPixels)
                _save("BS")
        finally:
            document.close()
            pass


if __name__ == "__main__":
    print(f"{bcolors.yellow('LastOrigin 2DModel Extractor')} by {bcolors.magenta('Wolfgang Kurz')}")
    print(f"{bcolors.darkyellow('- 2024-03-14-1a')}")
    print(f"{bcolors.blue(bcolors.underline('https://github.com/WolfgangKurz/LastOrigin-2DModel-Extractor'))}")

    while True:
        print("")
        file = input(f"{bcolors.green('2DModel File')} : ")

        if not os.path.exists(file):
            print(f"{bcolors.red('×')} File not found")
            continue

        try:
            main(file)
        except Exception as e:
            print(f"{bcolors.red('×')} Error on \"{sys.argv[1]}\"")
            print(e)
            print("")

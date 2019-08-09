# import filesio
import numpy as np
import cv2
import vtk

def show_img(data):
    if type(data) == type([]):
        num = len(data)
        for i in range(num):
            cv2.namedWindow(str(i), 0)
        for i in range(num):
            cv2.imshow(str(i), data[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif type(data) == type(np.array([])):
        cv2.imshow("image", data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_3D(all_cam_params, pointcloud=None, testparam=None, testinterface=None, testparam1=None, nonvisible=None):
    if pointcloud == None:
        colors = vtk.vtkNamedColors()
        # bkg = map(lambda x: x / 255.0, [])
        
        k = all_cam_params[0].getIntrinsic()

        joint = vtk.vtkAssembly()

        for i in all_cam_params:
            cube = vtk.vtkCubeSource()
            t = i.getMotion()
            cube.SetCenter(t)
            cube.SetXLength(0.5)
            cube.SetYLength(0.5)
            cube.SetZLength(0.5)
            cube.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(cube.GetOutput())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(colors.GetColor3d("Red"))

            joint.AddPart(actor)

        # display interface
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        for i, j, k in zip(testinterface[0], testinterface[1], testinterface[2]):
            id = points.InsertNextPoint(testparam1[:, i, j, k])
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d("Green"))

        joint.AddPart(actor)

        # display non-visible points
        if nonvisible != None:
            points = vtk.vtkPoints()
            vertices = vtk.vtkCellArray()
            for i, j, k in zip(nonvisible[0], nonvisible[1], nonvisible[2]):
                id = points.InsertNextPoint(testparam1[:, i, j, k])
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(id)

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetVerts(vertices)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(colors.GetColor3d("Red"))

            joint.AddPart(actor)
        # line = vtk.vtkLineSource()
        # line1 = vtk.vtkLineSource()
        # p0 = all_cam_params[17].getMotion()
        # p1 = testparam[0]
        # p2 = testparam[1]
        # print(p0, p1)
        # line.SetPoint1(p0)
        # line.SetPoint2(p1)
        # line1.SetPoint1(p0)
        # line1.SetPoint2(p2)
        #
        # linemapper = vtk.vtkPolyDataMapper()
        # linemapper.SetInputConnection(line.GetOutputPort())
        # lineactor = vtk.vtkActor()
        # lineactor.SetMapper(linemapper)
        # lineactor.GetProperty().SetLineWidth(0.5)
        # lineactor.GetProperty().SetColor(colors.GetColor3d("Black"))
        #
        # line1.SetPoint1(p0)
        # line1.SetPoint2(p2)
        # line1mapper = vtk.vtkPolyDataMapper()
        # line1mapper.SetInputConnection(line1.GetOutputPort())
        # line1actor = vtk.vtkActor()
        # line1actor.SetMapper(line1mapper)
        # line1actor.GetProperty().SetLineWidth(0.5)
        # line1actor.GetProperty().SetColor(colors.GetColor3d("Black"))
        #
        # joint.AddPart(lineactor)
        # joint.AddPart(line1actor)

        # cube = vtk.vtkCubeSource()
        # cube.SetCenter(testparam)
        # cube.SetXLength(0.5)
        # cube.SetYLength(0.5)
        # cube.SetZLength(0.5)
        # cube.Update()
        #
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputData(cube.GetOutput())
        #
        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        # actor.GetProperty().SetColor(colors.GetColor3d("Blue"))
        #
        # joint.AddPart(actor)

        # '''
        # display bounding cube
        joint.AddPart(construct_lineActor(testparam[0], testparam[1]))
        joint.AddPart(construct_lineActor(testparam[1], testparam[2]))
        joint.AddPart(construct_lineActor(testparam[2], testparam[3]))
        joint.AddPart(construct_lineActor(testparam[3], testparam[0]))
        joint.AddPart(construct_lineActor(testparam[0], testparam[4]))
        joint.AddPart(construct_lineActor(testparam[4], testparam[5]))
        joint.AddPart(construct_lineActor(testparam[5], testparam[6]))
        joint.AddPart(construct_lineActor(testparam[6], testparam[7]))
        joint.AddPart(construct_lineActor(testparam[1], testparam[5]))
        joint.AddPart(construct_lineActor(testparam[2], testparam[6]))
        joint.AddPart(construct_lineActor(testparam[3], testparam[7]))
        joint.AddPart(construct_lineActor(testparam[4], testparam[7]))
        # '''

        ren = vtk.vtkRenderer()
        ren.AddActor(joint)

        axes = vtk.vtkAxesActor()
        #  The axes are positioned with a user transform
        # axes.SetUserTransform(transform)
        axes.SetTotalLength([10] * 3)
        # default font size is 12
        axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(1)
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(1)
        axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(1)
        ren.AddActor(axes)
        ren.SetBackground(colors.GetColor3d("Silver"))

        renwin = vtk.vtkRenderWindow()
        renwin.SetSize(800, 800)
        renwin.AddRenderer(ren)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renwin)

        iren.Initialize()
        renwin.Render()
        iren.Start()
        # joint.AddPart(actor1)
    else:
        pass


# return an actor to show the frame
def construct_lineActor(p0, p1):
    colors = vtk.vtkNamedColors()
    line = vtk.vtkLineSource()
    line.SetPoint1(p0)
    line.SetPoint2(p1)
    
    linemapper = vtk.vtkPolyDataMapper()
    linemapper.SetInputConnection(line.GetOutputPort())
    lineactor = vtk.vtkActor()
    lineactor.SetMapper(linemapper)
    lineactor.GetProperty().SetLineWidth(1.5)
    lineactor.GetProperty().SetColor(colors.GetColor3d("Black"))

    return lineactor
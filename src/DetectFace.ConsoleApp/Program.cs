using DetectPeople.Face.Helpers;
using DetectPeople.Face.Retina;
using OpenCvSharp;
using System;
using System.Diagnostics;
using System.IO;

namespace DetectFace.ConsoleApp
{
    internal class Program
    {
        private static FaceDetectionAdv faceDetect;
        private static FaceRecognition faceRec;
        private static readonly Stopwatch timer = new();
        private const double FaceCoeficient = 0.65;

        static void Main(string[] args)
        {
            faceDetect = new FaceDetectionAdv();
            faceRec = new FaceRecognition();
            faceRec.Initialize();

            if (!Directory.Exists(PathHelper.ImagesFolder()))
            {
                Directory.CreateDirectory(PathHelper.ImagesFolder());
            }

            string filepath = "friends.jpg";

            timer.Restart();
            DetectFaces(filepath,"result.jpg");
            timer.Stop();
            Console.WriteLine($"DetectFaces done in {timer.ElapsedMilliseconds}ms.");
            Console.ReadKey();
        }


        private static void DetectFaces(string filePath, string fileName)
        {
            var detectedJunkFolder = "junk";
            if (!Directory.Exists(detectedJunkFolder))
            {
                Directory.CreateDirectory(detectedJunkFolder);
            }

            var faces = faceDetect.DetectFaces(filePath);
            Console.WriteLine($"{faces.Count} Faces Detected - {filePath}");

            foreach (var face in faces)
            {
                if (face.EyePointsScore < 0.75f)
                {
                    Console.WriteLine($"Eye score : {face.EyePointsScore}");
                    FaceDetectionAdv.SaveFaceImg(face, Cv2.ImRead(filePath), Path.Combine(detectedJunkFolder, $"eyescore_{Path.GetFileNameWithoutExtension(fileName)}_{Guid.NewGuid()}.jpg"));
                    continue;
                }


                if (!(IsNormal(face.Head.Yaw) && IsNormal(face.Head.Pitch) && IsNormal(face.Head.Roll)))
                {
                    Console.WriteLine($"Yaw : {Math.Round(face.Head.Yaw, 2)}, Pitch : {Math.Round(face.Head.Pitch, 2)}, Roll : {Math.Round(face.Head.Roll, 2)}");
                    FaceDetectionAdv.SaveFaceImg(face, Cv2.ImRead(filePath), Path.Combine(detectedJunkFolder, $"ypr_{Path.GetFileNameWithoutExtension(fileName)}_{Guid.NewGuid()}.jpg"));
                    continue;
                }

                if (!IsNormal(FaceDetectionAdv.Angle(face.Head.Axis[0])))
                {
                    Console.WriteLine("RED Head Axis > 20");
                    FaceDetectionAdv.SaveFaceImg(face, Cv2.ImRead(filePath), Path.Combine(detectedJunkFolder, $"red_{Path.GetFileNameWithoutExtension(fileName)}_{Guid.NewGuid()}.jpg"));
                    continue;
                }

                if (File.Exists(fileName))
                {
                    FaceDetectionAdv.SaveFaceImg(face, Cv2.ImRead(fileName).Clone(), fileName);
                }
                else 
                {
                    FaceDetectionAdv.SaveFaceImg(face, Cv2.ImRead(filePath), fileName);
                }                

                var faceRectImg = new Mat(face.Mat, face.FaceRectangle);
                var recognizedFaces = faceRec.Recognize(faceRectImg.Clone());
                var searchResult = recognizedFaces.Found;
                var best = recognizedFaces.Best;
                var coeficient = recognizedFaces.MaxDistance;
 
                if (coeficient > FaceCoeficient)
                {
                    Console.WriteLine($"{best.Label} : {coeficient}");
                    var personPath = Path.Combine(PathHelper.ImagesFolder(), best.Label, fileName);
                    SavePerson(filePath, faceRectImg, personPath, searchResult);
                }
                else // new person
                {
                    Console.WriteLine($"New person : {coeficient}");
                    var newPersonPath = Path.Combine(PathHelper.ImagesFolder(), "unknown_" + Guid.NewGuid().ToString(), fileName);
                    Directory.CreateDirectory(Path.GetDirectoryName(newPersonPath));

                    SavePerson(filePath, faceRectImg, newPersonPath, searchResult);
                }
            }
        }

        private static void SavePerson(string originalPath, Mat originalFile, string personPath, FaceRecognitionResult result)
        {
            string personFacePath = Path.ChangeExtension(personPath, ".png");
            var parts = personPath.Split(new char[] { '\\', '/' });
            var dir = parts[^2];

            originalFile.SaveImage(personFacePath);
            result.FilePath = personFacePath;
            result.Label = dir;
            result.Face = originalFile;

            File.Copy(originalPath, personPath, true);

            faceRec.AddFace(result);
        }

        private static bool IsNormal(float f)
        {
            return Math.Abs(f) <= 25;
        }
    }
}

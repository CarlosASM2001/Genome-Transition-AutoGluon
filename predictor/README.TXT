========================
BUILD OUTPUT DESCRIPTION
========================

When you build an Java application project that has a main class, the IDE
automatically copies all of the JAR
files on the projects classpath to your projects dist/lib folder. The IDE
also adds each of the JAR files to the Class-Path element in the application
JAR files manifest file (MANIFEST.MF).

To run the project from the command line, go to the dist folder and
type the following:

java -Xmx3024m -Xss9048k -cp ../../PredictorILP.jar util.PredictorILP
                                                            3 3 1 198295559 false true true 0 0 false MAPK-ERK true true false false -

Argumentos: 3 3 1 198295559 false true true 0 0 false MAPK-ERK true true false false -

args[0]: gffVersion (Versión del estándar GFF para generar los reportes de estructuras)(tipo entero)
args[1]: regionID   (Identificador del cromosoma desde donde son tomadas las secuencias que son procesadas)(tipo entero)
args[2]: regionInicio (Coordenada del inicio en la regionID en proceso)(tipo entero)
args[3]: regionFin    (Coordenada del fin en la regionID en proceso)(tipo entero)
args[4]: ilpinr	      (Se define si los sitios de inicio de trascrito se definen mediante uso de ILP o consenso INR)(true|false)
args[5]: consensos    (Establece si los core promoter se definen mediante consensos o factores de transcripción)(true|false)
args[6]: reporteAbs   (Se define si los reportes de estructuras se harán absolutos respecto de las coordenadas de Ensembl)(true|false)
args[7]: numObjs      (Corresponde al numero de objetos a traer desde PDB)(tipo entero)
args[8]: numIter      (Corresponde al numero de veces que debe irse a PDB)(tipo entero)
args[9]: ilpClasificador (Se define si los sitios de transición se definen mediante uso de ILP o clasificadores WEKA)(true|false)
args[10]: red (Define el nombre de la red de regulación asociada a la secuencia en proceso)(Alfanumérico)
args[11]: gorf (Define si se generarán estructuras de ORFs)
args[12]: gt (Define si se generarán estructuras de transcritos)
args[13]: gpr (Define si se generarán patrones de regulación)
args[14]: complementarios (Define si se generarán archivos complementarios relacionados con FTs presentes en las regiones de regulación)
args[15]: hebra (- Reverse | + Forward)

To distribute this project, zip up the dist folder (including the lib folder)
and distribute the ZIP file.

Notes:

* If two JAR files on the project classpath have the same name, only the first
JAR file is copied to the lib folder.
* Only JAR files are copied to the lib folder.
If the classpath contains other types of files or folders, these files (folders)
are not copied.
* If a library on the projects classpath also has a Class-Path element
specified in the manifest,the content of the Class-Path element has to be on
the projects runtime path.
* To set a main class in a standard Java project, right-click the project node
in the Projects window and choose Properties. Then click Run and enter the
class name in the Main Class field. Alternatively, you can manually type the
class name in the manifest Main-Class element.

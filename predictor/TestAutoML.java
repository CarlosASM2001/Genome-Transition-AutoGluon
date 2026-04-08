import clasificador.AutoMLClasificador;
import util.MiddleWare;

/**
 * Test simple para probar el clasificador AutoML.
 *
 * REQUISITO: El servicio FastAPI debe estar corriendo en http://143.198.77.77:8000
 *
 * Para iniciar el servicio:
 *   python -m uvicorn main:app --reload
 */
public class TestAutoML {

    public static void main(String[] args) {

        // Secuencia de ejemplo (la misma del curl que tienes)
        String secuencia = "ttcctagaccttatatgtctaaactggggcttcctgacataaaactatgcttaccggccaggaatctgttagaaaactcagagctcagtagaaggaacactggctttggaatgtgtgaggtctggttttgctcaaagtgtgcagtatgtgaaggagaacaatttactgaccattactctgccttactgattcaaattctgaggtttattgaataatttcttagattgccttccagctctaaatttctcagcaccaaaatgaagtccatttcaatctctctctctctctttccctcccgtacatatacacacactcatacatatatatggtcacaatagaaaggcaggtagatcagaagtctcagttgctgagaaagagggagggagggtgagccagaggtaccttctcccccattgtagagaaaagtgaagttcttttagagccccgttacatcttcaaggctttttatgagataatggaggaaataaagagggctcagtccttctactgtccatatttcattctcaaatctgttattagaggaatgattctgatctccacctaccatacacatgccctgttgcttgttgggccttcctaaaatgttagagtatgatgacagatggagttgtctgggtacatttgtgtgcatttaagggtgatagtgtatttgctctttaagagctgagtgtttgagcctctgtttgtgtgtaattgagt";

        System.out.println("================================================================");
        System.out.println("           TEST AUTOML - Predicción de sitios genómicos");
        System.out.println("================================================================\n");

        System.out.println("Secuencia: " + secuencia.length() + " nucleótidos");
        System.out.println("API: http://143.198.77.77:8000/predict\n");

        try {
            // FORMA 1: Usar directamente AutoMLClasificador
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            System.out.println("  FORMA 1: Uso directo de AutoMLClasificador");
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

            AutoMLClasificador clasificador = new AutoMLClasificador();

            System.out.println("Enviando request a FastAPI...");
            AutoMLClasificador.AutoMLResult resultado1 = clasificador.predict(secuencia);

            System.out.println("\n✓ Respuesta recibida:");
            System.out.println("  - Sitios EI: " + resultado1.getEiPositions().size());
            System.out.println("  - Sitios IE: " + resultado1.getIePositions().size());
            System.out.println("  - Sitios ZE: " + resultado1.getZePositions().size());
            System.out.println("  - Sitios EZ: " + resultado1.getEzPositions().size());

            System.out.println("\nPosiciones encontradas:");
            System.out.println("  EI: " + resultado1.getEiPositions());
            System.out.println("  IE: " + resultado1.getIePositions());
            System.out.println("  ZE: " + resultado1.getZePositions());
            System.out.println("  EZ: " + resultado1.getEzPositions());


            // FORMA 2: Usar a través de MiddleWare
            System.out.println("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            System.out.println("  FORMA 2: Uso a través de MiddleWare.predictAutoML()");
            System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

            MiddleWare middleware = new MiddleWare();

            System.out.println("Llamando a middleware.predictAutoML()...");
            AutoMLClasificador.AutoMLResult resultado2 = middleware.predictAutoML(secuencia);

            System.out.println("\n✓ Respuesta recibida:");
            System.out.println("  - Sitios EI: " + resultado2.getEiPositions().size());
            System.out.println("  - Sitios IE: " + resultado2.getIePositions().size());
            System.out.println("  - Sitios ZE: " + resultado2.getZePositions().size());
            System.out.println("  - Sitios EZ: " + resultado2.getEzPositions().size());


            System.out.println("\n\n================================================================");
            System.out.println("  ✓ TEST COMPLETADO EXITOSAMENTE");
            System.out.println("================================================================\n");

        } catch (Exception e) {
            System.err.println("\n================================================================");
            System.err.println("  ❌ ERROR EN EL TEST");
            System.err.println("================================================================\n");
            System.err.println("Error: " + e.getMessage());
            System.err.println("\nPosibles causas:");
            System.err.println("  1. El servicio FastAPI no está corriendo");
            System.err.println("  2. El servicio no está en http://143.198.77.77:8000");
            System.err.println("  3. Problema de conexión");
            System.err.println("\nSolución:");
            System.err.println("  Asegúrate de que el servicio esté corriendo:");
            System.err.println("    cd /ruta/al/servicio/fastapi");
            System.err.println("    python -m uvicorn main:app --reload");
            System.err.println("\nDetalles del error:");
            e.printStackTrace();
        }
    }
}

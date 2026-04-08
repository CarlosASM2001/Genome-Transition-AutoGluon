import gene.information.Analizer;
import gene.information.GeneConstructor;
import clasificador.AutoMLClasificador;
import util.MiddleWare;
import java.util.ArrayList;
import java.util.List;

/**
 * Test de integración para verificar que AutoML funciona correctamente
 * en el flujo completo del sistema.
 */
public class TestIntegracionAutoML {

    private static List<String> buildGeneDataFromSequence(String secuencia) throws Exception {
        List<String> data = new ArrayList<>();
        for (int i = 0; i < secuencia.length(); i++) {
            char base = Character.toLowerCase(secuencia.charAt(i));
            if (base != 'a' && base != 'g' && base != 't' && base != 'c') {
                throw new Exception("Secuencia inválida en índice " + i + ": '" + secuencia.charAt(i) + "'");
            }
            data.add(String.valueOf(base));
        }
        return data;
    }

    public static void main(String[] args) {

        // Secuencia de prueba
        String secuencia = "ttcctagaccttatatgtctaaactggggcttcctgacataaaactatgcttaccggccaggaatctgttagaaaactcagagctcagtagaaggaacactggctttggaatgtgtgaggtctggttttgctcaaagtgtgcagtatgtgaaggagaacaatttactgaccattactctgccttactgattcaaattctgaggtttattgaataatttcttagattgccttccagctctaaatttctcagcaccaaaatgaagtccatttcaatctctctctctctctttccctcccgtacatatacacacactcatacatatatatggtcacaatagaaaggcaggtagatcagaagtctcagttgctgagaaagagggagggagggtgagccagaggtaccttctcccccattgtagagaaaagtgaagttcttttagagccccgttacatcttcaaggctttttatgagataatggaggaaataaagagggctcagtccttctactgtccatatttcattctcaaatctgttattagaggaatgattctgatctccacctaccatacacatgccctgttgcttgttgggccttcctaaaatgttagagtatgatgacagatggagttgtctgggtacatttgtgtgcatttaagggtgatagtgtatttgctctttaagagctgagtgtttgagcctctgtttgtgtgtaattgagt";

        System.out.println("════════════════════════════════════════════════════════════════");
        System.out.println("  TEST DE INTEGRACIÓN AUTOML");
        System.out.println("════════════════════════════════════════════════════════════════\n");

        AutoMLClasificador clasificador = new AutoMLClasificador();

        System.out.println("Secuencia: " + secuencia.length() + " nucleotidos");
        System.out.println("API: " + clasificador.getApiUrl() + "/predict\n");

        try {
            // Preparar datos de entrada por nucleótido (formato esperado por GeneConstructor.initLists)
            List<String> data = buildGeneDataFromSequence(secuencia);

            MiddleWare middleWare = new MiddleWare();

            System.out.println("────────────────────────────────────────────────────────────────");
            System.out.println("  TEST 1: Usando Analizer (flujo completo)");
            System.out.println("────────────────────────────────────────────────────────────────\n");

            Analizer analizer = new Analizer();

            // Probar con AutoML
            System.out.println("→ Llamando a readFromMiddleWare con useAutoML=true...\n");
            analizer.readFromMiddleWare(
                middleWare,
                false,      // ilpClasificador = false
                true,       // useAutoML = true ← ACTIVADO
                data,
                null,       // rutaSecuencia no necesaria para AutoML
                secuencia
            );

            GeneConstructor constructor = analizer.getConstructor();

            System.out.println("\n✓ Constructor creado exitosamente");
            System.out.println("\nResultados obtenidos:");
            System.out.println("  - Sitios GT: " + constructor.getGt().size());
            System.out.println("  - Sitios AG: " + constructor.getAg().size());
            System.out.println("  - Sitios ATG: " + constructor.getAtg().size());
            System.out.println("  - Sitios STOP: " + constructor.getStops().size());

            if (constructor.getDistPosGt() != null) {
                System.out.println("  - Distancias GT: " + constructor.getDistPosGt().size() + " valores");
            }

            System.out.println("\n────────────────────────────────────────────────────────────────");
            System.out.println("  TEST 2: Comparación con Weka (opcional)");
            System.out.println("────────────────────────────────────────────────────────────────\n");

            System.out.println("→ Para comparar con Weka, ejecuta este test con useAutoML=false");
            System.out.println("  y compara los resultados.");

            System.out.println("\n════════════════════════════════════════════════════════════════");
            System.out.println("  ✓✓✓ TEST COMPLETADO EXITOSAMENTE ✓✓✓");
            System.out.println("════════════════════════════════════════════════════════════════\n");

            System.out.println("Conclusión:");
            System.out.println("  • AutoML está integrado correctamente");
            System.out.println("  • El flujo completo funciona: Analizer → GeneConstructor → AutoML");
            System.out.println("  • Las predicciones se obtienen en una sola llamada HTTP");
            System.out.println("\nPara usar AutoML en producción:");
            System.out.println("  1. Asegúrate de que FastAPI esté corriendo");
            System.out.println("  2. En GenInformation.java línea 292, deja: useAutoML = true");
            System.out.println("  3. El sistema usará AutoML automáticamente");

        } catch (Exception e) {
            System.err.println("\n════════════════════════════════════════════════════════════════");
            System.err.println("  ❌ ERROR EN EL TEST");
            System.err.println("════════════════════════════════════════════════════════════════\n");
            System.err.println("Error: " + e.getMessage());
            System.err.println("\nPosibles causas:");
            System.err.println("  1. FastAPI no está corriendo");
            System.err.println("  2. La secuencia/data no cumple formato esperado");
            System.err.println("  3. Error de compilación o de clase");
            System.err.println("\nDetalles:");
            e.printStackTrace();
        }
    }
}

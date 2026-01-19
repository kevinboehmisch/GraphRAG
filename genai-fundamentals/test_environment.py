import os
import unittest
from dotenv import load_dotenv, find_dotenv

# Lade Umgebungsvariablen
load_dotenv(find_dotenv())

class TestEnvironment(unittest.TestCase):

    skip_env_variable_tests = True
    skip_google_test = True   # Umbenannt von openai
    skip_neo4j_test = True

    def test_env_file_exists(self):
        # Prüft, ob .env da ist
        env_file_exists = True if find_dotenv() > "" else False
        if env_file_exists:
            TestEnvironment.skip_env_variable_tests = False
        self.assertTrue(env_file_exists, ".env file not found.")

    def env_variable_exists(self, variable_name):
        # Hilfsfunktion
        self.assertIsNotNone(
            os.getenv(variable_name),
            f"{variable_name} not found in .env file")

    def test_google_variables(self):
        # NEU: Prüft auf GOOGLE_API_KEY statt OpenAI
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping env variable test")

        self.env_variable_exists('GOOGLE_API_KEY')
        TestEnvironment.skip_google_test = False

    def test_neo4j_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping Neo4j env variables test")

        self.env_variable_exists('NEO4J_URI')
        self.env_variable_exists('NEO4J_USERNAME')
        self.env_variable_exists('NEO4J_PASSWORD')
        TestEnvironment.skip_neo4j_test = False

    def test_google_connection(self):
        # NEU: Testet die Verbindung zu Google Gemini
        if TestEnvironment.skip_google_test:
            self.skipTest("Skipping Google test")

        try:
            # Wir nutzen LangChain, da du das installiert hast
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # Kurzer Test-Aufruf
            llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-lite", 
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            response = llm.invoke("Hello, are you there?")
            connected = True if response else False
            
        except Exception as e:
            print(f"\n[FEHLER] Google Verbindung gescheitert: {e}")
            connected = False

        self.assertTrue(
            connected,
            "Google GenAI connection failed. Check GOOGLE_API_KEY in .env file.")

    def test_neo4j_connection(self):
        if TestEnvironment.skip_neo4j_test:
            self.skipTest("Skipping Neo4j connection test")

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USERNAME'), 
                  os.getenv('NEO4J_PASSWORD'))
        )
        try:
            driver.verify_connectivity()
            connected = True
        except Exception as e:
            print(f"\n[FEHLER] Neo4j Verbindung gescheitert: {e}")
            connected = False
        finally:
            driver.close()

        self.assertTrue(
            connected,
            "Neo4j connection failed. Check NEO4J_URI, USERNAME, PASSWORD in .env file."
        )

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnvironment('test_env_file_exists'))
    suite.addTest(TestEnvironment('test_google_variables')) # Geändert
    suite.addTest(TestEnvironment('test_neo4j_variables'))
    suite.addTest(TestEnvironment('test_google_connection')) # Geändert
    suite.addTest(TestEnvironment('test_neo4j_connection'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
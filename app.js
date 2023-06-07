import { OpenAI } from 'langchain/llms/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { HNSWLib } from 'langchain/vectorstores/hnswlib';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { loadQAStuffChain } from 'langchain/chains';
import { config } from 'dotenv';
import fs from 'fs';
config();

// Create an instance of the OpenAI model
const model = new OpenAI();
const sample = fs.readFileSync('./sample.txt', 'utf8');
const query = 'what is the color of the sky in autorama?';

// Create a text splitter object for dividing documents into smaller chunks.
// The splitter uses recursive character-based splitting technique.
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500, // The size of each chunk in characters
  chunkOverlap: 100, // The overlap size between consecutive chunks in characters
});

const main = async () => {
  try {
    // Create documents from the sample text using the text splitter
    const docs = await splitter.createDocuments([sample]);
    // Create a vector store from the documents using the OpenAI embeddings
    const vectorStore = await HNSWLib.fromDocuments(
      docs, // The documents to create the vector store from
      new OpenAIEmbeddings() // The embeddings to use for creating the vector store
    );
    // Perform a similarity search on the vector store using the query
    const result = await vectorStore.similaritySearch(query, 5);
    // Create a chain for loading the QA model and performing QA
    const chain = new loadQAStuffChain(model);
    // Perform QA on the query using the result of the similarity search
    const res = await chain.call({
      question: query, // The question to ask
      input_documents: result, // The documents to search for answers in
    });
    // Print the result
    console.log(res);
  } catch (error) {
    console.error(error);
  }
};

main();

import { OpenAI } from 'langchain/llms/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { HNSWLib } from 'langchain/vectorstores/hnswlib';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { loadQAStuffChain } from 'langchain/chains';
import { config } from 'dotenv';
import fs from 'fs';
config();

const model = new OpenAI();
const sample = fs.readFileSync('./sample.txt', 'utf8');
const query = 'what is the color of the sky in autorama?';

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100,
});

const main = async () => {
  try {
    const docs = await splitter.createDocuments([sample]);
    const vectorStore = await HNSWLib.fromDocuments(
      docs,
      new OpenAIEmbeddings()
    );
    const result = await vectorStore.similaritySearch(query, 5);
    const chain = new loadQAStuffChain(model);
    const res = await chain.call({
      question: query,
      input_documents: result,
    });
    console.log(res);
  } catch (error) {
    console.error(error);
  }
};

main();

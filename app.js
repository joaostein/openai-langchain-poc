import { Configuration, OpenAIApi } from "openai";
import { config } from "dotenv";
import fs from "fs";
config();

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);
const context = fs.readFileSync('./sample.txt', 'utf8');
const question = `What year can the roots of the Internet be traced back to?`;
const prompt = `${context}\nQ: ${question}`;

const main = async () => {
  try {
    const response = await openai.createCompletion({
      prompt: prompt,
      max_tokens: 32,
      temperature: 0,
      model: 'text-davinci-003',
    });
    console.log(response.data.choices[0].text);
  } catch (error) {
    console.error(error);
  }
};

main();
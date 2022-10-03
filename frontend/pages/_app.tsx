import "../styles/globals.css";
import type { AppProps } from "next/app";
import { ChakraProvider, HStack, VStack } from "@chakra-ui/react";
import app from "../firebase/firebase";
import { initializeApp } from "firebase/app";

// pages/_app.js

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <ChakraProvider>
      <VStack boxShadow="lg" minH="5xl" display="flex" w="2xl" marginX="auto">
        <Component {...pageProps} />
      </VStack>
    </ChakraProvider>
  );
}

export default MyApp;

// function MyApp({ Component, pageProps }: AppProps) {
//   return <Component {...pageProps} />
// }

// export default MyApp

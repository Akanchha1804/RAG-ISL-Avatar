# RAG-Augmented Text-to-ISL Avatar Generator

> **An AI-powered accessibility platform that converts speech or text
> into real-time Indian Sign Language using a 3D avatar.**

------------------------------------------------------------------------

## The Idea

Imagine speaking into your phone and watching a 3D avatar immediately
communicate your words in Indian Sign Language (ISL).

Instead of treating this as a simple language translation problem, the
system understands the sentence, retrieves similar ISL examples,
generates an ISL gloss sequence, and animates a digital avatar to
perform the signs.

The project focuses on creating a practical, real-time communication aid
for the Deaf and Hard-of-Hearing community while demonstrating modern AI
techniques such as Retrieval-Augmented Generation (RAG), transformer
models, vector search, and 3D animation.

------------------------------------------------------------------------

# The Problem

Most research and commercial systems work in one direction:

**Sign Language → Text**

Very few systems perform the reverse:

**Speech/Text → Indian Sign Language**

This is even more challenging for ISL because:

-   Public ISL datasets are relatively small.
-   ISL grammar differs from English.
-   Real-time avatar generation is still under-explored.

------------------------------------------------------------------------

# The Solution

The project combines speech recognition, retrieval, natural language
processing, and animation into one seamless pipeline.

``` text
User Speaks
      │
      ▼
Speech Recognition
      │
      ▼
English/Hindi Text
      │
      ▼
RAG Retrieval
      │
      ▼
Gloss Generation
      │
      ▼
Animation Mapping
      │
      ▼
3D ISL Avatar
```

------------------------------------------------------------------------

# How It Works

## 1. Speech Recognition

The user can either type a sentence or speak naturally.

Distil-Whisper converts speech into text.

Example:

> "I am going to school today."

------------------------------------------------------------------------

## 2. Retrieval-Augmented Generation (RAG)

Instead of translating immediately, the system first searches an ISL
knowledge base for similar sentence--gloss pairs.

These examples help the model understand how similar ideas are expressed
in Indian Sign Language.

This improves translation quality despite the limited size of available
ISL datasets.

------------------------------------------------------------------------

## 3. Gloss Generation

A fine-tuned sequence-to-sequence model generates an ISL gloss sequence
using:

-   The user's sentence
-   Retrieved examples
-   Learned ISL syntax

Example:

English

> I am going to school today.

Generated Gloss

``` text
TODAY
I
GO
SCHOOL
```

------------------------------------------------------------------------

## 4. Avatar Animation

Every gloss corresponds to a sign animation.

The animation engine looks up the required clips, blends transitions,
and plays them on a rigged 3D avatar to create smooth, understandable
signing.

------------------------------------------------------------------------

# Why RAG?

Traditional models rely only on what they learned during training.

This project first retrieves similar ISL examples before generating the
output.

Think of it like allowing the AI to consult a reference book before
answering a question.

Benefits:

-   Better translation consistency
-   Improved handling of low-resource datasets
-   More natural ISL gloss generation
-   Easier future expansion

------------------------------------------------------------------------

# Technology Stack

  Layer                Technology
  -------------------- --------------------------------
  Frontend             React + Tailwind CSS
  Backend              FastAPI
  Speech Recognition   Distil-Whisper
  Embeddings           sentence-transformers (MiniLM)
  Vector Database      FAISS
  Gloss Generator      Flan-T5 / mT5
  Deep Learning        PyTorch
  Avatar Engine        Unity
  Database             PostgreSQL
  Deployment           Docker

------------------------------------------------------------------------

# Project Highlights

-   Real-time speech-to-sign translation
-   Retrieval-Augmented Generation (RAG)
-   AI-powered ISL gloss generation
-   3D animated avatar
-   Accessibility-focused application
-   Modular architecture
-   Research-oriented design

------------------------------------------------------------------------

# Future Enhancements

-   Hindi support
-   Facial expressions and non-manual markers
-   Finger spelling for unknown words
-   Browser extension
-   Video-call integration
-   Mobile application
-   Larger ISL vocabulary

------------------------------------------------------------------------

# Final Vision

The goal is not just to build another AI demo.

The vision is to create a communication platform that helps bridge the
gap between spoken language and Indian Sign Language while showcasing
how retrieval, language models, and real-time animation can work
together to make technology more inclusive.
